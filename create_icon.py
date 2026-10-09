from PIL import Image, ImageDraw

def create_app_icon(output_path="app_icon.ico"):
    # Generate 256x256 icon
    size = (256, 256)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded rectangle background
    draw.rounded_rectangle([10, 10, 246, 246], radius=40, fill=(13, 17, 26, 255), outline=(0, 210, 255, 255), width=6)

    # Inner cyber reticle
    cx, cy = 128, 128
    draw.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], outline=(0, 255, 136, 200), width=4)
    draw.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], outline=(0, 210, 255, 255), width=5)

    # Center glowing pupil
    draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=(0, 210, 255, 255))
    draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(255, 255, 255, 255))

    # Corner brackets (bounding box aesthetics)
    bracket_col = (0, 255, 136, 255)
    bw = 6
    blen = 32
    # TL
    draw.line([(35, 35), (35 + blen, 35)], fill=bracket_col, width=bw)
    draw.line([(35, 35), (35, 35 + blen)], fill=bracket_col, width=bw)
    # TR
    draw.line([(221, 35), (221 - blen, 35)], fill=bracket_col, width=bw)
    draw.line([(221, 35), (221, 35 + blen)], fill=bracket_col, width=bw)
    # BL
    draw.line([(35, 221), (35 + blen, 221)], fill=bracket_col, width=bw)
    draw.line([(35, 221), (35, 221 - blen)], fill=bracket_col, width=bw)
    # BR
    draw.line([(221, 221), (221 - blen, 221)], fill=bracket_col, width=bw)
    draw.line([(221, 221), (221, 221 - blen)], fill=bracket_col, width=bw)

    # Save multi-size ICO
    img.save(
        output_path,
        format="ICO",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    )
    print(f"Icon successfully generated: {output_path}")

if __name__ == "__main__":
    create_app_icon()
