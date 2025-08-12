# Meeting Minutes Generator using LLaMA and Whisper

## 📌 Overview
This project is a **meeting minutes generator** that takes an audio recording (e.g., meeting, council session, podcast) and converts it into **structured markdown notes** with summaries, key discussion points, takeaways, and action items with owners.  

It combines **OpenAI’s Whisper** for speech-to-text transcription and **Meta’s LLaMA 3.1 Instruct** model for natural language processing, making it a powerful end-to-end automation tool for documentation.

The solution works entirely in a **local development environment** (VS Code, terminal, or server) without relying on Google Colab, but the original Colab version is kept for reference in the code.

---

## 🎯 Use Cases
- **Corporate Meeting Notes** – Automatically generate professional summaries of team meetings.
- **Council & Committee Transcripts** – Turn raw recordings into structured action reports.
- **Podcast Summaries** – Extract key points and highlights from long-form audio.
- **Lecture Notes** – Convert class recordings into concise, readable study material.
- **Customer Support Calls** – Summarize discussions for record-keeping and follow-up.

---

## ⚙️ How It Works
1. **Audio Input** – The user provides an `.mp3` file of the meeting.
2. **Transcription** – The audio is transcribed to plain text using **OpenAI Whisper** (`whisper-1` model).
3. **Prompt Construction** – A system message and user prompt are created to instruct the LLaMA model on how to structure the meeting minutes.
4. **Text Generation** – The **Meta LLaMA 3.1 Instruct** model processes the transcript and generates well-formatted markdown output.
5. **Streaming Output** – Results are streamed live in the terminal as they are generated, avoiding long waits.
6. **Final Output** – The completed minutes can be saved, shared, or integrated into documentation systems.

---

## 🔑 Requirements
You will need:
- **OpenAI API Key** – for Whisper transcription.
- **Hugging Face Access Token** – for downloading and running the LLaMA model.

### Obtaining Keys
- **OpenAI API Key**: Sign up at [OpenAI](https://platform.openai.com/), go to your account settings, and create a new API key.
- **Hugging Face Token**: Sign up at [Hugging Face](https://huggingface.co/), go to “Settings” → “Access Tokens”, and generate a read token.

---

## 🚀 Implementation Notes
- The project supports **4-bit quantization with BitsAndBytes** to optimize memory usage and allow large model inference on consumer GPUs.
- The script uses the **`TextStreamer`** feature from `transformers` to display output in real time.
- Both the Colab and VS Code-compatible versions of the code are included in a single file for flexibility.
- Environment variables are used to store API keys securely, ensuring no sensitive data is hardcoded.

---

## 📄 Summary
This project automates the tedious task of manually writing meeting minutes. By integrating **speech-to-text** and **large language models**, it produces highly structured, human-readable summaries in seconds. It’s adaptable, secure, and efficient — ideal for anyone who needs accurate documentation from spoken content.
