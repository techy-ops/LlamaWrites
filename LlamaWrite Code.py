import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from diffusers import StableDiffusionPipeline
import torch

# Set Streamlit page configuration
st.set_page_config(page_title="AI Blog Generator")

# Ensure offload folder exists for large models
offload_folder = "./offload"
os.makedirs(offload_folder, exist_ok=True)

# Use GPT-2 model for text generation
model_name = "gpt2"  # Public GPT-2 model

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_text_generation_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            offload_folder=offload_folder,
            low_cpu_mem_usage=True
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading text generation model: {e}")
        return None

# Load GPT-2 text generation model
generator = load_text_generation_model()

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_stable_diffusion_model():
    try:
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        st.write(f"Stable Diffusion model loaded on {device.upper()}.")
        return pipe
    except Exception as e:
        st.error(f"Error loading image generation model: {e}")
        return None

# Load Stable Diffusion model
stable_diffusion_model = load_stable_diffusion_model()

def generate_blog_content(topic, word_count):
    with st.spinner("Generating blog content..."):
        try:
            prompt = f"Write a detailed blog about {topic}. This tool is a game-changer for content creators, researchers, and professionals."
            max_tokens = min(word_count * 7, 1024)  # GPT-2's max token length is 1024
            result = generator(prompt, max_length=max_tokens, do_sample=True)
            return result[0]['generated_text']
        except Exception as e:
            st.error(f"Error generating content: {e}")
            return None

def generate_image_from_topic(blog_content):
    with st.spinner("Generating related image..."):
        try:
            prompt = f"An artistic illustration representing: {blog_content[:100]}..."
            image = stable_diffusion_model(prompt).images[0]
            return image
        except Exception as e:
            st.error(f"Error generating image: {e}")
            return None

def main():
    st.title("AI Blog Generator")

    topic = st.text_input("Enter Blog Topic:")
    word_count = st.number_input("Enter desired word count:", min_value=50, max_value=5000, value=1000)

    if topic and st.button("Generate Blog and Image"):
        if generator:
            blog_content = generate_blog_content(topic, word_count)
            if blog_content:
                st.subheader("Generated Blog Post:")
                st.write(blog_content)
                st.write(f"Word Count: {len(blog_content.split())}")

                # Generate image based on the blog content
                image = generate_image_from_topic(blog_content)
                if image:
                    st.subheader("Related Image:")
                    st.image(image, use_column_width=True)
                else:
                    st.warning("Could not generate an image for this topic.")
            else:
                st.warning("Blog content was not generated.")
        else:
            st.error("Failed to load the model. Please try again.")

if __name__ == "__main__":
    main()
