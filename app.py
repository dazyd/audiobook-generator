import streamlit as st
import os
import time
import wave
import struct
import json
from pypdf import PdfReader
from pydub import AudioSegment
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ---------------------------------------------------------
# GOOGLE DRIVE SETUP (SERVICE ACCOUNT)
# ---------------------------------------------------------
def get_drive_service(service_account_info):
    """Authenticates Drive using Service Account JSON credentials."""
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

def get_or_create_drive_folder(service, folder_name="Audiobook"):
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)", supportsAllDrives=True).execute()
    items = results.get('files', [])
    if items:
        return items[0]['id']
    meta = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    folder = service.files().create(body=meta, fields='id', supportsAllDrives=True).execute()
    return folder.get('id')

def upload_file_to_drive(service, file_path, folder_id, mime_type='audio/wav'):
    file_name = os.path.basename(file_path)
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
    uploaded_file = service.files().create(
        body=file_metadata, 
        media_body=media, 
        fields='id',
        supportsAllDrives=True
    ).execute()
    return uploaded_file.get('id')

# ---------------------------------------------------------
# TEXT & AUDIO HELPERS
# ---------------------------------------------------------
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        return uploaded_file.read().decode("utf-8")

def split_into_chunks(text, max_words=200):
    paragraphs = text.split("\n")
    chunks, current, count = [], [], 0
    for p in paragraphs:
        words = p.strip().split()
        if not words:
            continue
        if count + len(words) <= max_words:
            current.append(p.strip())
            count += len(words)
        else:
            if current:
                chunks.append("\n".join(current))
            current = [p.strip()]
            count = len(words)
    if current:
        chunks.append("\n".join(current))
    return chunks

def parse_audio_mime(mime_type: str) -> dict:
    bits, rate = 16, 24000
    if mime_type:
        for param in mime_type.split(";"):
            p = param.strip()
            if p.lower().startswith("rate="):
                try: rate = int(p.split("=", 1)[1])
                except: pass
            elif p.startswith("audio/L"):
                try: bits = int(p.split("L", 1)[1])
                except: pass
    return {"bits_per_sample": bits, "rate": rate}

def convert_pcm_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    p = parse_audio_mime(mime_type)
    bits = p["bits_per_sample"]
    rate = p["rate"]
    data_size = len(audio_data)
    block_align = (bits // 8)
    byte_rate = rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ",
        16, 1, 1, rate, byte_rate, block_align, bits, b"data", data_size
    )
    return header + audio_data

def merge_audio_chunks(file_paths, output_path, output_format):
    combined = AudioSegment.empty()
    for fp in file_paths:
        segment = AudioSegment.from_wav(fp)
        combined += segment
    combined.export(output_path, format=output_format)

# ---------------------------------------------------------
# UI SETUP & API KEY VAULT
# ---------------------------------------------------------
st.set_page_config(page_title="AI Audiobook Studio", layout="wide")
st.title("🎙️ Gemini Native Audiobook Generator")

VAULT_FILE = "api_keys_vault.txt"
saved_keys = ""
if os.path.exists(VAULT_FILE):
    with open(VAULT_FILE, "r") as f:
        saved_keys = f.read()

# Sidebar: API Key Manager Notes
with st.sidebar:
    st.header("🔑 API Keys Vault (Notes)")
    keys_input = st.text_area(
        "Paste all your AI Studio API Keys here (One key per line):", 
        value=saved_keys,
        height=200,
        placeholder="AIzaSy...\nAIzaSy..."
    )
    if st.button("💾 Save Keys to Vault"):
        with open(VAULT_FILE, "w") as f:
            f.write(keys_input.strip())
        st.success("Keys saved successfully!")
        st.rerun()

    key_list = [k.strip() for k in keys_input.split("\n") if k.strip()]
    selected_api_key = ""
    if key_list:
        selected_api_key = st.selectbox("Select Active API Key:", key_list)
    else:
        st.caption("No keys saved yet. Paste keys above and click Save.")

# Main Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Credentials & Files")
    api_key = st.text_input("Active Gemini API Key", value=selected_api_key, type="password")
    uploaded_file = st.file_uploader("2. Upload TXT or PDF Document", type=["txt", "pdf"])
    
    enable_drive = st.checkbox("1. Save to Google Drive ('Audiobook' folder)", value=True)
    drive_key_file = None
    if enable_drive:
        drive_key_file = st.file_uploader("Upload Service Account JSON Key", type=["json"])

    st.subheader("Model Parameters")
    model_name = st.text_input("7. Model Name", value="gemini-3.1-flash-tts-preview")
    voice_name = st.text_input("4. Voice Name (Manual)", value="Charon")
    chunk_size = st.number_input("3. Chunk Size (words)", min_value=50, max_value=500, value=200)
    temperature = st.slider("5. Temperature", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
    base_delay = st.number_input("6. Base Delay Between Chunks (seconds)", min_value=0, max_value=120, value=60)

with col2:
    st.subheader("🎭 Audio Profile & Scene")
    scene = st.text_area("8. Scene", value="The Corporate Studio: A quiet, acoustically treated narrative recording space.")
    sample_context = st.text_area(
        "9. Sample Context & Style",
        value="Instructional E-learning. Measured pacing with clear pauses for clarity. Tone is authoritative, accessible, and articulate."
    )
    output_format = st.selectbox("10. Output Audio Format", ["wav", "mp3"])

# ---------------------------------------------------------
# EXECUTION & VISUAL PROCESSING LOG
# ---------------------------------------------------------
if st.button("🚀 Start Audiobook Production", type="primary"):
    if not api_key:
        st.error("API Key missing! Sidebar me keys save karein ya manually enter karein.")
        st.stop()
    if not uploaded_file:
        st.error("Kripya TXT ya PDF file upload karein!")
        st.stop()

    drive_service = None
    folder_id = None
    if enable_drive:
        if drive_key_file is not None:
            try:
                service_info = json.load(drive_key_file)
                drive_service = get_drive_service(service_info)
                folder_id = get_or_create_drive_folder(drive_service, "Audiobook")
                st.sidebar.success("✅ Google Drive Connected!")
            except Exception as e:
                st.error(f"Google Drive Error: {str(e)}")
                st.stop()
        else:
            st.error("Google Drive connect karne ke liye JSON key upload karein ya checkbox uncheck karein.")
            st.stop()

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        )
    )

    full_text = extract_text(uploaded_file)
    chunks = split_into_chunks(full_text, max_words=chunk_size)
    total_chunks = len(chunks)
    base_name = os.path.splitext(uploaded_file.name)[0]

    os.makedirs("temp_chunks", exist_ok=True)
    generated_wav_paths = []

    st.markdown("---")
    st.subheader("📊 Live Processing Status")
    
    progress_bar = st.progress(0)
    current_status_box = st.empty()
    countdown_box = st.empty()
    log_container = st.container(height=320)

    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        chunk_file = f"temp_chunks/{base_name}_chunk_{i}.wav"

        # 1. Processing Step
        current_status_box.info(f"▶️ Chunk {chunk_num}/{total_chunks} processing...")
        log_container.write(f"▶️ Chunk {chunk_num}/{total_chunks} processing...")

        if not os.path.exists(chunk_file):
            prompt = f"""Read the transcript according to these instructions:

# Scene
{scene}

# Style Context
{sample_context}

# Transcript
{chunk}"""

            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            
            success = False
            while not success:
                try:
                    response = client.models.generate_content(
                        model=model_name, contents=contents, config=config
                    )
                    raw_bytes = bytearray()
                    mime_type = "audio/L16;rate=24000"
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if part.inline_data and part.inline_data.data:
                                raw_bytes.extend(part.inline_data.data)
                                if part.inline_data.mime_type:
                                    mime_type = part.inline_data.mime_type

                    if len(raw_bytes) > 500:
                        wav_data = convert_pcm_to_wav(bytes(raw_bytes), mime_type)
                        with open(chunk_file, "wb") as f:
                            f.write(wav_data)
                        success = True
                    else:
                        time.sleep(base_delay)
                except Exception as e:
                    log_container.error(f"⚠️ Error on Chunk {chunk_num}: {str(e)}")
                    time.sleep(base_delay)

        generated_wav_paths.append(chunk_file)

        # 2. Drive Upload Step
        if drive_service and folder_id:
            current_status_box.info(f"⏳ Chunk {chunk_num}/{total_chunks} saving into Google Drive...")
            log_container.write(f"⏳ Chunk {chunk_num}/{total_chunks} saving into Google Drive...")
            try:
                upload_file_to_drive(drive_service, chunk_file, folder_id, mime_type="audio/wav")
                log_container.write(f"✅ Chunk {chunk_num} saved")
            except Exception as e:
                log_container.error(f"❌ Chunk {chunk_num} drive upload failed: {str(e)}")
        else:
            log_container.write(f"✅ Chunk {chunk_num} saved locally")

        log_container.write("")
        progress_bar.progress(chunk_num / total_chunks)

        if chunk_num < total_chunks and base_delay > 0:
            for s in range(base_delay, 0, -1):
                countdown_box.caption(f"⏳ Waiting for API rate limit: {s} seconds remaining...")
                time.sleep(1)
            countdown_box.empty()

    # 3. Chunks Done
    st.success("🎉 All chunks processed and saved successfully.")
    log_container.write("🎉 All chunks processed and saved successfully.")

    # 4. Merging Step
    current_status_box.info("🔄 Merging into a full Audiobook...")
    log_container.write("🔄 Merging into a full Audiobook...")
    
    final_output_name = f"{base_name}_Audiobook.{output_format}"
    merge_audio_chunks(generated_wav_paths, final_output_name, output_format)

    if drive_service and folder_id:
        try:
            drive_mime = "audio/mpeg" if output_format == "mp3" else "audio/wav"
            upload_file_to_drive(drive_service, final_output_name, folder_id, mime_type=drive_mime)
        except Exception as e:
            log_container.error(f"❌ Master file drive upload failed: {str(e)}")

    # 5. Success
    current_status_box.empty()
    st.balloons()
    st.success("✅ All files successfully merged")
    log_container.write("✅ All files successfully merged")

    st.audio(final_output_name)
    with open(final_output_name, "rb") as f:
        st.download_button("📥 Download Master Audiobook", f, file_name=final_output_name)
