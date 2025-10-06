# Algorithm Video Generator

Create algorithm explanation videos with Dr. Doofenshmirtz's voice over Minecraft gameplay.

## Quick Start

1. **Install Python 3.8+** and **FFmpeg**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add required files:**
   ```
   backgrounds/minecraft.mp4     # Gameplay video
   voices/dr_doof.wav           # Voice sample
   images/doof_excited.png      # Character images
   images/doof_shocked.png
   images/doof_explaining.png
   images/doof_smug.png
   ```

4. **Run the app:**
   ```bash
   python app.py
   ```

5. **Open browser:** http://localhost:7860

## How to Use

Write your script with timestamps and moods:

```
[0-5] [excited] Welcome to binary search!
[5-10] [explaining] It divides the array in half each time
[10-15] [smug] That's O(log n) complexity!
```

**Available moods:** excited, shocked, explaining, smug

## Requirements

- 8GB+ RAM
- GPU recommended for speed
- 10GB free space for models

## Troubleshooting

- **Out of memory:** Make shorter videos
- **Files not found:** Check file paths match exactly
- **Slow processing:** Use GPU, close other apps
- **First run is slow:** Downloads 2GB AI models

That's it! The app handles everything else automatically.
```

**Mac:**
```bash
brew install ffmpeg
```

### 5. Setup Required Files
Create the following directory structure and add your files:

```
brain-rot-app/
├── app.py
├── requirements.txt
├── backgrounds/
│   └── minecraft.mp4          # Your gameplay footage
├── voices/
│   └── dr_doof.wav           # Dr. Doof voice sample
├── images/
│   ├── doof_excited.png      # Doof excited expression
│   ├── doof_shocked.png      # Doof shocked expression
│   ├── doof_explaining.png   # Doof explaining expression
│   └── doof_smug.png         # Doof smug expression
├── fonts/ (optional)
│   └── Lexend-Bold.ttf       # Custom font
└── output/                   # Generated videos will be saved here
```

## Usage

### 1. Start the Application
```bash
python app.py
```

### 2. Open Web Interface
- Open your browser and go to `http://localhost:7860`
- The interface will load automatically

### 3. Create Your Video
1. **Enter Video Title**: Give your video a descriptive title
2. **Write Script**: Use the timestamp and mood format:
   ```
   [0-5] [excited] Welcome to my algorithm explanation!
   [5-12] [explaining] Here's how sliding window works
   [12-18] [smug] Easy right? Follow for more tips!
   ```
3. **Click Generate**: Wait for processing (first run takes longer for model download)

### Script Format

**Timestamp Format:** `[start-end]` in seconds
**Mood Options:**
- `excited` - High energy, enthusiastic
- `shocked` - Surprised, dramatic reaction  
- `explaining` - Teaching mode, gesturing
- `smug` - Confident, "I told you so"

**Example:**
```
[0-3] [excited] Today we're learning binary search!
[3-8] [explaining] It works by dividing the search space in half
[8-12] [shocked] Wait, that's only O(log n) time complexity!
[12-15] [smug] Pretty clever, right? Subscribe for more!
```

## Performance Tips

### GPU Acceleration
- Ensure CUDA is properly installed
- Monitor GPU memory usage during processing
- Close other GPU-intensive applications

### Processing Speed
- Keep video segments between 3-10 seconds for optimal pacing
- Use shorter scripts for faster processing
- SSD storage significantly improves performance

### Memory Management
- The app automatically manages GPU memory
- Restart if you encounter memory errors
- Consider shorter videos for systems with limited RAM

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce video length
- Close other applications
- Restart the application

**"File not found" errors**
- Check all required files are in correct directories
- Verify file paths and extensions
- Ensure FFmpeg is installed and in PATH

**TTS model download fails**
- Check internet connection
- Ensure sufficient disk space (2GB+ for models)
- Try running again (downloads can be resumed)

**Video encoding errors**
- Install/update FFmpeg
- Check system codec support
- Try CPU encoding if GPU encoding fails

### Performance Issues

**Slow processing**
- Enable GPU acceleration
- Use SSD storage
- Reduce video complexity
- Close background applications

**Poor video quality**
- Check source video resolution
- Verify FFmpeg installation
- Adjust encoding parameters in code

## File Requirements

### Voice Sample (dr_doof.wav)
- Format: WAV, MP3, or FLAC
- Length: 10-30 seconds recommended
- Quality: Clear audio, minimal background noise
- Content: Natural speech sample of target voice

### Background Video (minecraft.mp4)
- Format: MP4, AVI, or MOV
- Resolution: 1080p or higher recommended
- Duration: 5+ minutes for variety
- Content: Engaging gameplay footage

### Character Images (PNG format)
- Resolution: 800x800 or similar
- Background: Transparent PNG preferred
- Style: Consistent art style across moods
- Quality: High resolution for crisp overlay

## Advanced Configuration

### GPU Settings
Edit `app.py` to modify GPU behavior:
```python
# Adjust worker count for your system
max_workers = min(4, multiprocessing.cpu_count())

# Modify memory management
torch.cuda.empty_cache()
```

### Video Quality
Modify encoding parameters:
```python
# Adjust quality (lower = better, range: 18-28)
'-cq', '23',

# Adjust bitrate
'-b:v', '8M',
```

## License

This project is for educational purposes. Ensure you have rights to all media files used.

## Support

For issues and questions:
1. Check this README for solutions
2. Verify all requirements are met
3. Check console output for error details
4. Ensure all files are properly formatted and located
