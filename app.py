
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# 1. Configuration
# =====================================

st.set_page_config(
    page_title="Cat vs Dog Classification",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱 Cat vs Dog Classification")
st.write("Upload an image to classify it as a Cat or Dog.")

IMG_SIZE = 180

# Make sure this matches training class order
CLASS_NAMES = ["Cat", "Dog"]


# =====================================
# 2. Load CNN Model
# =====================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cnn_dog_cat.keras"
    )

try:
    model = load_model()
    st.success("CNN Model loaded successfully!")

except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# =====================================
# 3. Upload Image
# =====================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# =====================================
# 4. Image Preprocessing
# =====================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Resize image
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # Convert image to NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # IMPORTANT:
        # Do not divide by 255 here.
        # The model already contains Rescaling(1./255).

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # =================================
        # 5. Prediction
        # =================================

        if st.button("Predict Image", type="primary"):

            with st.spinner("Analyzing image..."):

                prediction = model.predict(
                    image_array,
                    verbose=0
                )

            # Binary classification (Sigmoid)
            if prediction.shape[-1] == 1:

                probability = float(prediction[0][0])

                predicted_index = (
                    1 if probability >= 0.5 else 0
                )

                confidence = (
                    probability
                    if predicted_index == 1
                    else 1 - probability
                )

            # Multiclass classification (Softmax)
            else:

                predicted_index = int(
                    np.argmax(prediction[0])
                )

                confidence = float(
                    prediction[0][predicted_index]
                )

            # Get predicted class
            result = CLASS_NAMES[predicted_index]

            # =================================
            # 6. Display Results
            # =================================

            st.divider()

            st.subheader("Prediction Result")

            if result == "Cat":
                st.success("🐱 Predicted Class: Cat")
            else:
                st.success("🐶 Predicted Class: Dog")

            st.metric(
                label="Confidence Score",
                value=f"{confidence * 100:.2f}%"
            )

            st.progress(float(confidence))

            # Show raw prediction
            st.caption(
                f"Raw model output: {float(prediction[0][0]):.4f}"
                if prediction.shape[-1] == 1
                else f"Predicted probability: {confidence:.4f}"
            )

    except Exception as e:
        st.error(f"Image processing error: {e}")
