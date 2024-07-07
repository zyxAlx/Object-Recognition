import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label, filedialog
from threading import Thread
from PIL import Image, ImageTk
from stop_sign_detector import StopSignDetector
from stop_sign_annotator import StopSignAnnotator


class MainApp:
    def __init__(self, cascade_path):
        self.video_path = None
        self.cap = None
        self.detector = StopSignDetector(cascade_path)
        self.annotator = StopSignAnnotator()

        self.width = 640  # Default width, will be updated after video selection
        self.height = 480  # Default height, will be updated after video selection
        self.new_width = int(self.width * 0.5)
        self.new_height = int(self.height * 0.5)

        self.playing = False
        self.paused = False

        self.root = tk.Tk()
        self.root.title("Stop Sign Detector")
        self.root.geometry(f"{self.new_width + 20}x{self.new_height + 100}")

        # Create and center video frame
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(expand=True, fill=tk.BOTH)

        # Place video inside frame
        self.video_label = Label(self.video_frame)
        self.video_label.pack(pady=(10, 0))

        # Load icons
        self.play_icon = ImageTk.PhotoImage(file='images/play_icon.png')
        self.pause_icon = ImageTk.PhotoImage(file='images/pause_icon.png')
        self.open_folder_icon = ImageTk.PhotoImage(file='images/open-folder.png')


        # Add control buttons
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=(0, 10))

        self.select_button = tk.Button(self.control_frame, image=self.open_folder_icon, command=self.select_video)
        self.select_button.pack(side=tk.LEFT)

        self.play_button = tk.Button(self.control_frame, image=self.play_icon, command=self.play_video, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT)

        self.pause_button = tk.Button(self.control_frame, image=self.pause_icon, command=self.pause_video, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT)

        self.backward_button = tk.Button(self.control_frame, text="<< 3s", command=self.skip_backward, state=tk.DISABLED)
        self.backward_button.pack(side=tk.LEFT)

        self.forward_button = tk.Button(self.control_frame, text=">> 3s", command=self.skip_forward, state=tk.DISABLED)
        self.forward_button.pack(side=tk.LEFT)



        self.update_thread = Thread(target=self.update_window_position)
        self.update_thread.daemon = True
        self.update_thread.start()

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.new_width = int(self.width * 0.5)
            self.new_height = int(self.height * 0.5)
            self.root.geometry(f"{self.new_width + 20}x{self.new_height + 100}")
            self.play_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            self.backward_button.config(state=tk.NORMAL)
            self.forward_button.config(state=tk.NORMAL)

    def play_video(self):
        if not self.playing:
            self.playing = True
            self.paused = False
            Thread(target=self.run).start()
        else:
            self.paused = False

    def pause_video(self):
        self.paused = True

    def skip_forward(self):
        self.skip_video(3)

    def skip_backward(self):
        self.skip_video(-3)

    def skip_video(self, seconds):
        if self.cap:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip_frames = int(seconds * fps)
            new_frame = current_frame + skip_frames

            # Ensure the new frame position is within video boundaries
            if new_frame < 0:
                new_frame = 0
            elif new_frame >= total_frames:
                new_frame = total_frames - 1

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

    def quit_app(self):
        self.playing = False
        self.paused = True
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

    def run(self):
        while self.cap.isOpened() and self.playing:
            if self.paused:
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (self.new_width, self.new_height))
            frame_area = self.new_width * self.new_height
            stop_signs = self.detector.detect(resized_frame)
            self.annotator.annotate(resized_frame, stop_signs, frame_area, self.detector)

            # Convert frame to ImageTk format
            img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)

            # Update video with new image
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_window_position(self):
        previous_x = self.root.winfo_x()
        previous_y = self.root.winfo_y()
        while self.playing:
            current_x = self.root.winfo_x()
            current_y = self.root.winfo_y()
            if current_x != previous_x or current_y != previous_y:
                previous_x = current_x
                previous_y = current_y
            self.root.after(10)

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    cascade_path = 'train/cascade_stop_sign.xml'
    app = MainApp(cascade_path)
    app.start()