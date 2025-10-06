import gradio as gr
import os
import re
from pathlib import Path
from TTS.api import TTS
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
    concatenate_videoclips
)
from moviepy.video.fx.resize import resize
import torch
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gc
import cv2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


# Check if CUDA is available and setup
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Optimize CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"Using device: {device} (GPU not available)")

# Hardcoded Doof image mappings
DOOF_IMAGES = {
    "excited": "images/doof_excited.png",
    "shocked": "images/doof_shocked.png",
    "explaining": "images/doof_explaining.png",
    "smug": "images/doof_smug.png"
}

# Global TTS model
tts = None

def init_tts():
    """Initialize TTS model with voice cloning and GPU optimization"""
    global tts
    if tts is None:
        print("Loading TTS model (this may take a few minutes on first run)...")
        # Use XTTS v2 for voice cloning with GPU optimization
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        # Enable GPU optimizations if available
        if device == "cuda":
            print("Enabling GPU optimizations for TTS...")
            # Keep model in FP32 for numerical stability during sampling
            # XTTS requires precise probability calculations
            
            # Optimize memory usage
            torch.cuda.empty_cache()
            
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            
            # Set model to eval mode for inference optimizations
            tts.synthesizer.tts_model.eval()
        
        print("TTS model loaded successfully!")
    return tts

def parse_script(script_text):
    """
    Parse script format:
    [0-5] [excited] Text for this segment
    [5-10] [explaining] More text here
    
    Moods: excited, shocked, explaining, smug
    """
    segments = []
    lines = script_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for timestamp and optional mood marker [start-end] [mood] text
        pattern = r'\[(\d+)-(\d+)\]\s*(?:\[(\w+)\])?\s*(.+)'
        match = re.match(pattern, line)
        
        if match:
            start, end, mood, text = match.groups()
            
            # Default to explaining if no mood specified
            if not mood or mood.lower() not in DOOF_IMAGES:
                mood = "explaining"
            
            segment = {
                'start': int(start),
                'end': int(end),
                'mood': mood.lower(),
                'text': text.strip()
            }
            segments.append(segment)
    
    return segments

def generate_voiceover(text, output_path, voice_sample_path):
    """Generate voice using TTS with voice cloning and GPU acceleration"""
    tts_model = init_tts()
    
    print(f"Generating voice for: {text[:50]}...")
    
    # Clear GPU cache before inference
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Generate speech with voice cloning
    # Use inference_mode for speed without autocast (FP32 is needed for stability)
    with torch.inference_mode():
        tts_model.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=voice_sample_path,
            language="en",
            speed=1.0
        )
    
    # Clear GPU cache after inference
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return output_path

def create_text_image_gpu_optimized(text, size=(1080, 400), font_size=55, highlight_word_index=None):
    """GPU-optimized text image creation with OpenCV acceleration - FIXED positioning and highlighting"""
    # Use OpenCV for faster image operations if available
    try:
        import cv2
        # Create image using OpenCV (faster than PIL for large operations)
        img = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        
        # Convert to PIL for text rendering (still need PIL for fonts)
        pil_img = Image.fromarray(img, 'RGBA')
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load Lexend Bold font from the fonts folder, with fallbacks
        font_paths = [
            "fonts/Lexend-Bold.ttf",
            "Lexend-Bold.ttf",
            "Lexend-Bold",
            "LexendDeca-Bold.ttf",
            "lexend-bold.ttf",
        ]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception as e:
                continue

        if font is None:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("calibri.ttf", font_size)
                except:
                    font = ImageFont.load_default()

        # Replace problematic Unicode characters with ASCII equivalents
        text = text.replace('—', '-').replace('–', '-').replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

        # Word wrap text with more conservative width to prevent cutoff
        words = text.split()
        lines = []
        current_line = []
        word_positions = []  # Track which line each word is on

        for word_idx, word in enumerate(words):
            test_line = ' '.join(current_line + [word])
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                # Reduced from 100 to 120 for more padding to prevent cutoff
                if bbox[2] - bbox[0] < size[0] - 120:  # 60px padding on each side
                    current_line.append(word)
                    word_positions.append(len(lines))
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = [word]
                    word_positions.append(len(lines))
            except UnicodeEncodeError:
                # If there's still a Unicode error, skip this word or replace it
                safe_word = word.encode('ascii', 'ignore').decode('ascii')
                if safe_word:
                    current_line.append(safe_word)
                    word_positions.append(len(lines))

        if current_line:
            lines.append(current_line)

        # Calculate total text height with proper line spacing
        line_height = font_size + 25  # Increased spacing between lines
        total_height = len(lines) * line_height

        # Position text at top of canvas with some padding (not 82% which pushes it too low)
        y = 50  # Start 50px from top of the text canvas for padding

        # Draw each line
        word_counter = 0
        for line_idx, line_words in enumerate(lines):
            line_text = ' '.join(line_words)

            # Get line bbox safely
            try:
                bbox = draw.textbbox((0, 0), line_text, font=font)
                text_width = bbox[2] - bbox[0]
            except UnicodeEncodeError:
                # Fallback width calculation
                text_width = len(line_text) * (font_size * 0.6)

            # Center horizontally
            x = (size[0] - text_width) // 2
            current_x = x

            # Draw each word in the line
            for word_idx, word in enumerate(line_words):
                is_highlighted = (highlight_word_index is not None and word_counter == highlight_word_index)

                # Get word bbox safely with better measurements
                try:
                    # Measure from actual baseline position
                    word_bbox = draw.textbbox((0, 0), word, font=font)
                    word_width = word_bbox[2] - word_bbox[0]
                    word_height = word_bbox[3] - word_bbox[1]
                    # Get descent for proper vertical alignment
                    ascent, descent = font.getmetrics()
                except UnicodeEncodeError:
                    # Fallback dimensions
                    word_width = len(word) * (font_size * 0.6)
                    word_height = font_size
                    ascent = font_size
                    descent = 0
                except AttributeError:
                    # Fallback if getmetrics not available
                    word_width = word_bbox[2] - word_bbox[0] if 'word_bbox' in locals() else len(word) * (font_size * 0.6)
                    word_height = word_bbox[3] - word_bbox[1] if 'word_bbox' in locals() else font_size
                    ascent = font_size
                    descent = font_size * 0.2

                # If highlighted → draw pill background with proper coverage
                if is_highlighted:
                    pad_x, pad_y = 20, 15  # Even more padding for full coverage
                    
                    # Calculate proper vertical bounds - use actual text bbox for accuracy
                    # Get the actual bounding box of the text to ensure full coverage
                    try:
                        actual_bbox = draw.textbbox((current_x, y), word, font=font)
                        rect_y1 = max(0, actual_bbox[1] - pad_y)  # Top of text minus padding
                        rect_y2 = min(size[1], actual_bbox[3] + pad_y)  # Bottom of text plus padding
                    except:
                        # Fallback if bbox fails
                        rect_y1 = max(0, y - pad_y)
                        rect_y2 = min(size[1], y + word_height + pad_y)
                    
                    # Calculate horizontal bounds
                    rect_x1 = max(0, current_x - pad_x)
                    rect_x2 = min(size[0], current_x + word_width + pad_x)

                    draw.rounded_rectangle(
                        [rect_x1, rect_y1, rect_x2, rect_y2],
                        radius=20,  # Larger radius for smoother appearance
                        fill="#F8423C"
                    )

                # Draw black outline for better contrast
                outline_range = 3
                for adj_x in range(-outline_range, outline_range + 1):
                    for adj_y in range(-outline_range, outline_range + 1):
                        if adj_x != 0 or adj_y != 0:
                            try:
                                draw.text((current_x + adj_x, y + adj_y), word, font=font, fill="black")
                            except UnicodeEncodeError:
                                pass  # Skip outline if Unicode error

                # Draw word text in #F6F6DB
                try:
                    draw.text((current_x, y), word, font=font, fill="#F6F6DB")
                except UnicodeEncodeError:
                    # Try with ASCII-safe version
                    safe_word = word.encode('ascii', 'ignore').decode('ascii')
                    if safe_word:
                        draw.text((current_x, y), safe_word, font=font, fill="#F6F6DB")

                # Add word spacing
                current_x += word_width + int(font_size * 0.25)  # Slightly increased word spacing
                word_counter += 1

            y += line_height

        return np.array(pil_img)
    
    except ImportError:
        # Fallback to original implementation
        return create_text_image(text, size, font_size, highlight_word_index)

# Alias the optimized function
create_text_image = create_text_image_gpu_optimized

def estimate_word_timings(text, total_duration):
    """Estimate when each word should be highlighted based on speech duration"""
    words = text.split()
    if not words:
        return []
    
    # Simple estimation: distribute time evenly across words
    # In a real implementation, you might use speech recognition or phoneme timing
    word_timings = []
    time_per_word = total_duration / len(words)
    
    for i, word in enumerate(words):
        start_time = i * time_per_word
        end_time = (i + 1) * time_per_word
        word_timings.append({
            'word': word,
            'start': start_time,
            'end': end_time,
            'index': i
        })
    
    return word_timings

def create_karaoke_text_clip_parallel(text, duration, video_size=(1080, 1920)):
    """Parallel processing version of karaoke text clip creation"""
    from moviepy.editor import ImageSequenceClip
    
    word_timings = estimate_word_timings(text, duration)
    fps = 30
    total_frames = int(duration * fps)
    
    # Limit parallel workers to prevent memory overload
    max_workers = min(4, multiprocessing.cpu_count())  # Reduced from 8 to 4
    
    print(f"  → Creating {total_frames} frames with {max_workers} workers...")
    
    def create_frame(frame_num):
        current_time = frame_num / fps
        highlighted_word_index = None
        for timing in word_timings:
            if timing['start'] <= current_time < timing['end']:
                highlighted_word_index = timing['index']
                break
        
        return create_text_image(
            text, 
            size=(video_size[0], 500),  # Increased height to 500 to prevent cutoff
            highlight_word_index=highlighted_word_index
        )
    
    # Use parallel processing for frame creation with progress
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = list(executor.map(create_frame, range(total_frames)))
    
    print(f"  → Frames generated, creating clip...")
    
    if frames:
        clip = ImageSequenceClip(frames, fps=fps)
        # Position in lower third: 1200 pixels from top (about 62% down for 1920 height)
        clip = clip.set_position(('center', 1200))
        clip = clip.set_duration(duration)
        return clip
    else:
        text_img = create_text_image(text, size=(video_size[0], 500))
        from moviepy.editor import ImageClip
        clip = ImageClip(text_img).set_duration(duration)
        clip = clip.set_position(('center', 1200))
        return clip

# Replace the original function
create_karaoke_text_clip = create_karaoke_text_clip_parallel

def load_and_preprocess_background(gameplay_path, video_width=1080, video_height=1920):
    """GPU-accelerated background video loading and preprocessing"""
    background = VideoFileClip(gameplay_path)
    
    # Use OpenCV for faster video operations if possible
    bg_aspect = background.w / background.h
    target_aspect = video_width / video_height
    
    if bg_aspect > target_aspect:
        new_width = int(background.h * target_aspect)
        x_center = background.w / 2
        background = background.crop(x1=x_center - new_width/2, x2=x_center + new_width/2)
    else:
        new_height = int(background.w / target_aspect)
        y_center = background.h / 2
        background = background.crop(y1=y_center - new_height/2, y2=y_center + new_height/2)
    
    background = background.resize((video_width, video_height))
    return background

def generate_video(title, script, progress=gr.Progress()):
    """Main video generation function with GPU acceleration"""
    try:
        progress(0, desc="Initializing...")
        
        # Clear GPU memory at start
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Create output folder with title
        if not title or not title.strip():
            title = "untitled_video"
        
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
        safe_title = safe_title.strip().replace(' ', '_')
        
        # Paths
        gameplay_path = "backgrounds/minecraft.mp4"
        voice_sample = "voices/dr_doof.wav"
        output_dir = Path("output") / safe_title
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if files exist
        if not os.path.exists(gameplay_path):
            return None, "Error: Gameplay video not found at backgrounds/minecraft.mp4"
        
        if not os.path.exists(voice_sample):
            return None, "Error: Voice sample not found at voices/dr_doof.wav"
        
        missing_images = [mood for mood, path in DOOF_IMAGES.items() if not os.path.exists(path)]
        if missing_images:
            return None, f"Error: Missing Doof images: {', '.join(missing_images)}\nMake sure these files exist in the images/ folder:\n" + "\n".join([f"- {path}" for path in DOOF_IMAGES.values()])
        
        # Parse script
        progress(0.1, desc="Parsing script...")
        segments = parse_script(script)
        
        if not segments:
            return None, "Error: No valid segments found in script.\nFormat: [0-5] [excited] Your text here\nMoods: excited, shocked, explaining, smug"
        
        print(f"Found {len(segments)} segments")
        
        # Load background video with GPU optimization
        progress(0.2, desc="Loading gameplay footage...")
        video_width = 1080
        video_height = 1920
        background = load_and_preprocess_background(gameplay_path, video_width, video_height)
        
        # Pre-generate all voiceovers in parallel to better utilize GPU
        progress(0.3, desc="Generating voiceovers...")
        audio_paths = []
        
        for i, segment in enumerate(segments):
            audio_path = output_dir / f"voice_{i}.wav"
            generate_voiceover(segment['text'], str(audio_path), voice_sample)
            audio_paths.append(audio_path)
            
            # Progress update
            progress(0.3 + (i / len(segments)) * 0.3, desc=f"Generated voiceover {i+1}/{len(segments)}")
        
        # Create video clips with optimized processing
        progress(0.6, desc="Creating video clips...")
        video_clips = []
        
        for i, segment in enumerate(segments):
            print(f"\n=== Processing clip {i+1}/{len(segments)} ===")
            text = segment['text']
            mood = segment['mood']
            
            # Load audio
            print(f"Loading audio for segment {i+1}...")
            audio = AudioFileClip(str(audio_paths[i]))
            actual_duration = audio.duration
            print(f"Audio duration: {actual_duration:.2f}s")
            
            # Get background clip segment
            print(f"Extracting background segment...")
            bg_start = random.uniform(0, max(0, background.duration - actual_duration - 5))
            bg_clip = background.subclip(bg_start, bg_start + actual_duration)
            
            # Create karaoke text overlay (now parallelized)
            print(f"Creating karaoke text animation ({int(actual_duration * 30)} frames)...")
            text_clip = create_karaoke_text_clip(text, actual_duration, (video_width, video_height))
            print(f"Text animation complete!")
            
            # Load Doof image
            print(f"Loading Doof image ({mood})...")
            doof_image_path = DOOF_IMAGES[mood]
            doof_clip = ImageClip(doof_image_path)
            
            # Optimize Doof image processing
            max_img_width = 800
            max_img_height = 800
            
            doof_aspect = doof_clip.w / doof_clip.h
            if doof_clip.w > max_img_width or doof_clip.h > max_img_height:
                if doof_aspect > 1:
                    doof_clip = doof_clip.resize(width=max_img_width)
                else:
                    doof_clip = doof_clip.resize(height=max_img_height)
            
            doof_clip = doof_clip.set_position(('center', 200))
            doof_clip = doof_clip.set_duration(actual_duration)
            
            # Composite video with overlays
            print(f"Compositing layers...")
            final_clip = CompositeVideoClip([bg_clip, doof_clip, text_clip])
            final_clip = final_clip.set_audio(audio)
            
            video_clips.append(final_clip)
            print(f"✓ Clip {i+1}/{len(segments)} complete!")
            
            progress(0.6 + (i / len(segments)) * 0.2, desc=f"Created clip {i+1}/{len(segments)}")
            
            # Clear memory after each segment
            if device == "cuda":
                torch.cuda.empty_cache()
        
        # Concatenate all segments
        progress(0.8, desc="Combining segments...")
        print(f"\n=== Concatenating {len(video_clips)} segments ===")
        final_video = concatenate_videoclips(video_clips, method="compose")
        print(f"✓ Concatenation complete! Total duration: {final_video.duration:.2f}s")
        
        # Export final video with GPU-optimized settings
        progress(0.9, desc="Exporting video...")
        print(f"\n=== Exporting final video ===")
        output_path = output_dir / "final_video.mp4"
        
        # Try GPU-accelerated encoding with correct color space parameters
        try:
            print("Attempting NVENC GPU encoding...")
            ffmpeg_params = [
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',      # Fast preset for NVENC (p1-p7, p4 is balanced)
                '-tune', 'hq',        # High quality
                '-rc', 'vbr',         # Variable bitrate
                '-cq', '23',          # Quality level (lower = better, 23 is good)
                '-b:v', '8M',         # Target bitrate
                '-maxrate', '12M',    # Max bitrate
                '-bufsize', '16M',    # Buffer size
                '-pix_fmt', 'yuv420p',  # Proper pixel format
                '-colorspace', 'bt709',  # Correct color space
                '-color_primaries', 'bt709',
                '-color_trc', 'bt709',
                '-color_range', 'tv',  # TV range (16-235) for compatibility
            ]
            
            final_video.write_videofile(
                str(output_path),
                fps=30,
                codec='h264_nvenc',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                threads=8,
                ffmpeg_params=ffmpeg_params
            )
            print("✓ NVENC encoding successful!")
            
        except Exception as codec_error:
            print(f"GPU encoding failed: {codec_error}")
            print("Falling back to CPU encoding...")
            # Fallback to CPU with optimizations
            final_video.write_videofile(
                str(output_path),
                fps=30,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                preset='medium',  # medium is good balance for CPU
                threads=multiprocessing.cpu_count()
            )
            print("✓ CPU encoding complete!")
        
        # Cleanup
        background.close()
        for clip in video_clips:
            clip.close()
        
        # Final GPU cleanup
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        progress(1.0, desc="Complete!")
        
        gpu_info = f"\nGPU Info: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}"
        
        return str(output_path), f"✅ Video generated successfully!\n\nTitle: {title}\nOutput folder: output/{safe_title}/\nDuration: {final_video.duration:.1f}s\nSegments: {len(segments)}{gpu_info}\n\nOptimizations applied:\n- GPU-accelerated TTS inference (FP32)\n- Parallel text rendering\n- NVENC hardware encoding\n- CUDA memory optimization\n- Fixed text positioning (higher)\n- Improved highlight coverage\n\nFiles saved:\n- final_video.mp4\n- voice_0.wav to voice_{len(segments)-1}.wav"
        
    except Exception as e:
        # Clear GPU memory on error
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Algorithm Video Generator") as app:
        gr.Markdown("# Algorithm Video Generator with Doof")
        gr.Markdown("Create engaging algorithm explanation videos with Doof's voice and expressions!")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Video Details")
                
                title_input = gr.Textbox(
                    label="Video Title",
                    placeholder="Sliding Window Algorithm",
                    lines=1
                )
                
                gr.Markdown("### Script with Mood Instructions")
                gr.Markdown("""
                **Format:**
                ```
                [0-5] [excited] Introduction text here
                [5-12] [explaining] Explanation of the algorithm
                [12-18] [smug] Conclusion and call to action
                ```
                
                **Available Moods:**
                - `excited` - High energy, enthusiastic (with gadget)
                - `shocked` - Surprised, dramatic reaction
                - `explaining` - Teaching mode (gesturing)
                - `smug` - Confident, "I told you so"
                
                If no mood specified, defaults to `explaining`.
                """)
                
                script_input = gr.Textbox(
                    label="Script with Timestamps and Moods",
                    placeholder="[0-3] [excited] Sliding Window technique explained\n[3-7] [explaining] Two pointers - left shrinks, right expands\n[7-10] [smug] O(n) time - follow for more!",
                    lines=15
                )
                
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### Output")
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Textbox(label="Status", lines=5)
        
        gr.Markdown("""
        ### Required Files:
        - `backgrounds/minecraft.mp4` - Gameplay footage
        - `voices/dr_doof.wav` - Voice sample
        - `images/doof_excited.png` - Doof excited expression
        - `images/doof_shocked.png` - Doof shocked expression
        - `images/doof_explaining.png` - Doof explaining expression
        - `images/doof_smug.png` - Doof smug expression
        - `fonts/Lexend-Bold.ttf` - Lexend Bold font (optional, will fallback to system fonts)
        
        ### Tips:
        - Keep segments 3-10 seconds for best pacing
        - Match moods to the energy of your text
        - Use `excited` for hooks and reveals
        - Use `explaining` for main content
        - Use `smug` or `shocked` for conclusions
        - First run downloads TTS models (~2GB, takes 5-10 min)
        - Text features karaoke-style word highlighting
        - Model runs in FP32 for stability
        """)
        
        generate_btn.click(
            fn=generate_video,
            inputs=[title_input, script_input],
            outputs=[video_output, status_output]
        )
    
    return app

if __name__ == "__main__":

    if os.name == 'nt':
        import asyncio
        import warnings
        import logging
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)

    print("Starting Algorithm Video Generator...")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("\nMake sure your files are in place:")
    print("  - backgrounds/minecraft.mp4")
    print("  - voices/dr_doof.wav")
    print("  - images/doof_excited.png")
    print("  - images/doof_shocked.png")
    print("  - images/doof_explaining.png")
    print("  - images/doof_smug.png")
    print("\nLaunching interface...")
    
    app = create_interface()
    app.launch(share=False)