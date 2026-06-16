import streamlit as st
import torch
from src.model_loader import load_model
from src.inference import predict

MODEL_NAME = "Vu74269/bert-squadv2-extend-base"

st.set_page_config(
    page_title="QA App",
    page_icon="🤖",
    layout="centered"
)

st.title("Question Answering App v2.0")
st.write("Using fine-tuned BERT base uncased model on a modified SQuAD 2.0 dataset")


# Load model
try:
    tokenizer, model = load_model()
    st.success("Model loaded successfully")

except Exception as e:
    st.error("Failed to load model")
    st.exception(e)
    st.stop()


# Input fields
context = st.text_area(
    "Context",
    height=300,
    placeholder="Paste context here..."
)

question = st.text_input(
    "Question",
    placeholder="Ask a question..."
)

# Inference
if st.button("Ask"):
    if not context.strip():
        st.warning("Please enter context")
        st.stop()

    if not question.strip():
        st.warning("Please enter question")
        st.stop()

    with st.spinner("Finding answer..."):
        try:
            result = predict(
                tokenizer,
                model,
                question,
                context,
            )

            answer = result["answer"].strip()
            score = float(result["score"])

            # Debug output
            # st.divider()
            # st.write("### Debug Output")
            # st.json(result)

            # Display answer
            if answer == "":
                st.warning("No answer found in the context")
            else:
                st.success(answer)

            # Confidence score
            st.write("### Confidence")
            st.write(f"{score:.6f}")

        except Exception as e:
            st.error("Inference failed")
            st.exception(e)

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.write(f"Model path:")
st.sidebar.code(MODEL_NAME)

if torch.cuda.is_available():
    st.sidebar.success("GPU detected")
else:
    st.sidebar.info("Running on CPU")

# Footer
st.divider()
st.caption(
    "Fine-tuned Question Answering model using SQuAD 2.0"
)
