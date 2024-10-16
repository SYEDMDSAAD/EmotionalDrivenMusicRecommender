import streamlit as st
import cv2
import numpy as np
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from recommendations import recommend

def load_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights('model.h5')
    return model

def main():
    # Load the pre-trained model
    model = load_model()

    # Mapping the original emotion indices to broader categories
    emotion_dict = {
        0: "Angry", 1: "Angry", 2: "Calm",
        3: "Happy", 4: "Calm", 5: "Sad", 6: "Happy"
    }

    # Ensure session state for critical variables
    if 'emotion_list' not in st.session_state:
        st.session_state.emotion_list = []

    st.title("Emotion-based music recommendation")
    st.markdown(f"Welcome {st.session_state.username}!")

    col1 , col2, col3 = st.columns(3)

    with col2:
        if st.button('SCAN EMOTION(Click here)'):
            cap = cv2.VideoCapture(0)
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                count += 1
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                    prediction = model.predict(cropped_img)
                    max_index = int(np.argmax(prediction))

                    # Append detected emotion to session state
                    st.session_state.emotion_list.append(emotion_dict[max_index])

                    cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

                if cv2.waitKey(1) & 0xFF == ord('s') or count >= 20:
                    break

            cap.release()
            cv2.destroyAllWindows()
            st.success("Emotions successfully detected", icon="✅")

    if st.session_state.emotion_list:
        st.subheader("Detected Emotions:")

        # Count occurrences of each emotion
        emotion_counts = Counter(st.session_state.emotion_list)

        # Display the counts of each emotion
        for emotion, count in emotion_counts.items():
            st.write(f"{emotion}: {count} times")

        # Get the most common emotion
        most_common_emotion, _ = emotion_counts.most_common(1)[0]
        st.subheader("Most Predominant Emotion: " + most_common_emotion)

        # Use the most commonly detected emotion for recommendations
        st.subheader("Recommended Songs for Emotion: " + most_common_emotion)

        recommended_songids = recommend(st.session_state.sp, most_common_emotion)
        for song_id in recommended_songids:
            song = st.session_state.sp.track(song_id)
            song_info = f"[{song['name']}]({song['external_urls']['spotify']}) by {', '.join(artist['name'] for artist in song['artists'])}"
            st.write(song_info)  # Display song name as a clickable link
    else:
        st.write("No emotions detected.")

if __name__ == "__main__":
    main()