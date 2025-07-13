import gradio as gr
from elevenlabs import ElevenLabs
from blog_summarizer import summarize_blog, sanitize_filename
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Directory to store generated podcast files
AUDIO_DIR = "podcasts"
os.makedirs(AUDIO_DIR, exist_ok=True)

def process_url(url):
    try:
        # Step 1: Generating blog summary
        summary = summarize_blog(url)
        print("🔍 Blog Summary Generated.")

        # Step 2: Generating audio from summary
        try:
            client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

            audio = client.text_to_speech.convert(
                text=summary[:350],
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_flash_v2_5",
                output_format="mp3_44100_128",
            )

            filename = sanitize_filename(url).replace(".md", ".mp3")
            filepath = os.path.join(AUDIO_DIR, filename)

            with open(filepath, "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            return "✅ Podcast generated successfully!", summary, filepath

        except Exception as audio_error:
            print(f"🎧 Audio generation failed: {audio_error}")
            return "⚠️ Summary generated, but audio failed.", summary, None

    except Exception as e:
        return f"❌ Error: {str(e)}", "", None

# Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ AI Podcast Generator")
    gr.Markdown("Enter a blog URL to generate a podcast episode from its content.")
    
    with gr.Row():
        url_input = gr.Textbox(label="🔗 Blog URL", placeholder="https://example.com/blog-post")
    
    generate_btn = gr.Button("🎧 Generate Podcast")

    with gr.Row():
        status_output = gr.Textbox(label="📝 Status", lines=1)
    
    with gr.Row():
        summary_output = gr.Textbox(
            label="📄 Blog Summary",
            lines=25,
            max_lines=50,
            show_copy_button=True
        )

    with gr.Row():
        audio_output = gr.Audio(label="🎧 Podcast Audio", type="filepath")

    generate_btn.click(
        fn=process_url,
        inputs=[url_input],
        outputs=[status_output, summary_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7777, share=True)