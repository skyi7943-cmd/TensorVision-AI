import cv2
import numpy as np

def generate_demo_video(filename="demo_video.mp4", duration_sec=5, fps=30):
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    total_frames = duration_sec * fps

    for frame_idx in range(total_frames):
        # Dark road background
        frame = np.ones((h, w, 3), dtype=np.uint8) * 35

        # Draw road
        cv2.rectangle(frame, (0, 200), (w, 400), (55, 55, 60), -1)
        # Road lane markings
        dash_offset = (frame_idx * 8) % 40
        for x in range(-40 + dash_offset, w + 40, 40):
            cv2.line(frame, (x, 300), (x + 20, 300), (200, 200, 200), 2)

        # Draw moving car (simple stylized car)
        car_x = (frame_idx * 6) % (w + 120) - 100
        car_y = 240
        # Car body
        cv2.rectangle(frame, (car_x, car_y + 15), (car_x + 90, car_y + 40), (40, 100, 220), -1)
        cv2.rectangle(frame, (car_x + 20, car_y), (car_x + 70, car_y + 20), (50, 120, 240), -1)
        # Windows
        cv2.rectangle(frame, (car_x + 25, car_y + 3), (car_x + 42, car_y + 16), (220, 230, 240), -1)
        cv2.rectangle(frame, (car_x + 46, car_y + 3), (car_x + 65, car_y + 16), (220, 230, 240), -1)
        # Wheels
        cv2.circle(frame, (car_x + 20, car_y + 40), 9, (20, 20, 20), -1)
        cv2.circle(frame, (car_x + 70, car_y + 40), 9, (20, 20, 20), -1)
        cv2.circle(frame, (car_x + 20, car_y + 40), 4, (180, 180, 180), -1)
        cv2.circle(frame, (car_x + 70, car_y + 40), 4, (180, 180, 180), -1)

        # Title
        cv2.putText(frame, "TensorVision AI Offline Video Test Clip", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_idx:03d} / {total_frames}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Generated demo video: {filename} ({total_frames} frames)")

if __name__ == "__main__":
    generate_demo_video()
