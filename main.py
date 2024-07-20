import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans

class RubiksCube:
    def __init__(self):
        # Initialize a solved cube state using facelet notation
        self.state = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

    def set_state(self, state_str):
        # Set the cube state from a string
        if len(state_str) == 54:
            self.state = state_str
        else:
            raise ValueError("Invalid cube state string length. Expected 54 characters.")

    def get_state(self):
        # Get the current cube state as a string
        return self.state

    def identify_colors_kmeans(self, frame, n_colors=6):
        pixels = frame.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        color_names = ['red', 'green', 'blue', 'yellow', 'orange', 'white']
        color_map = {tuple(color): name for color, name in zip(colors, color_names)}

        height, width = frame.shape[:2]
        cell_height, cell_width = height // 3, width // 3
        identified_colors = []

        for i in range(3):
            for j in range(3):
                y1, y2 = i * cell_height, (i + 1) * cell_height
                x1, x2 = j * cell_width, (j + 1) * cell_width
                cell = frame[y1:y2, x1:x2]
                cell_pixels = cell.reshape(-1, 3)
                cell_labels = kmeans.predict(cell_pixels)
                dominant_color = colors[np.argmax(np.bincount(cell_labels))]
                identified_colors.append(color_map[tuple(dominant_color)])

        return identified_colors

    def update_cube_state(self, colors, face_index):
        # Update the cube state using identified colors for a specific face
        facelet_indices = self.get_facelet_indices(face_index)
        for i, idx in enumerate(facelet_indices):
            self.state = self.state[:idx] + colors[i][0][0] + self.state[idx+1:]

    def get_facelet_indices(self, face_index):
        # Returns the list of facelet indices for a given face index
        facelet_indices = []
        base_index = face_index * 9
        for i in range(3):
            for j in range(3):
                facelet_indices.append(base_index + i * 3 + j)
        return facelet_indices

def identify_rubiks_cube_color_in_the_frame(cube):
    st.set_page_config(page_title="Rubik's Cube Color Identifier", layout="wide")
    st.title("Rubik's Cube Color Identifier")
    st.caption("Powered by OpenCV, Streamlit, and K-means clustering")

    if 'current_face' not in st.session_state:
        st.session_state.current_face = 0
    if 'cube_faces' not in st.session_state:
        st.session_state.cube_faces = [None] * 6

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Capturing Cube Faces")
        frame_placeholder = st.empty()
        if st.session_state.current_face < 6:
            capture_button = st.button("Capture")

    with col2:
        st.subheader("Cube State")
        cube_state_display = st.empty()

        for i in range(6):
            if st.session_state.cube_faces[i] is not None:
                cube_state_display.write(f"Face {i + 1}: {st.session_state.cube_faces[i]}")

    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not capture_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image. Please try again.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

    if 'capture_button' in locals() and capture_button:
        face_colors = cube.identify_colors_kmeans(frame)
        cube.update_cube_state(face_colors, st.session_state.current_face)
        st.session_state.cube_faces[st.session_state.current_face] = ''.join(face_colors)
        st.session_state.current_face = (st.session_state.current_face + 1) % 6

    cap.release()
    st.experimental_rerun()

if __name__ == "__main__":
    cube = RubiksCube()
    identify_rubiks_cube_color_in_the_frame(cube)