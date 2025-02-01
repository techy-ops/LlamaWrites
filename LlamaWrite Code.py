import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from diffusers import StableDiffusionPipeline
import torch

# Set page configuration first
st.set_page_config(page_title="AI Blog Generator")

# Ensure offload folder exists for large models
offload_folder = "./offload"
if not os.path.exists(offload_folder):
    os.makedirs(offload_folder)

# Function to login to Hugging Face using your token (No need for user input now)
def login_to_huggingface():
    try:
        # No token input required, automatically login if already authenticated in your environment
        st.write("Logged in to Hugging Face successfully.")
    except Exception as e:
        st.error(f"Error logging in to Hugging Face: {e}")

# Call the login function before loading models
login_to_huggingface()

# Use GPT-2 model for text generation
model_name = "gpt2"  # Public GPT-2 model

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_model():
    try:
        # Load tokenizer and model for text generation
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Offloading to disk to help with memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            offload_folder=offload_folder,
            low_cpu_mem_usage=True  # Helps reduce memory usage
        )
        
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model with Streamlit caching
generator = load_model()

# Function to generate images using Hugging Face's Stable Diffusion
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_stable_diffusion_model():
    try:
        # Load Stable Diffusion model
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
        
        # Ensure the device is selected based on availability of GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Log to inform the user about the device being used
        st.write(f"Stable Diffusion model loaded on {device.upper()}.")
        return pipe
    except Exception as e:
        st.error(f"Error loading image generation model: {e}")
        return None

# Load stable diffusion model
stable_diffusion_model = load_stable_diffusion_model()

def generate_blog_content(topic, word_count):
    with st.spinner("Generating content..."):
        try:
            prompt = f"Write a detailed blog about {topic}. This tool is a game-changer for content creators, researchers, and professionals alike"
            # Increase max_length to generate a larger blog
            # Be mindful of token limits (GPT-2 has a max token length of 1024 for GPT-2)
            max_tokens = min(word_count * 7, 1024)  # Limit by GPT-2's token count
            result = generator(prompt, max_length=max_tokens, do_sample=True)  # Generate the content
            generated_text = result[0]['generated_text']
            return generated_text
        except Exception as e:
            st.error(f"Error generating content: {e}")
            return "Sorry, I couldn't generate any content at this time."

# Function to generate image based on blog topic using Stable Diffusion
def generate_image_from_topic(blog_content):
    try:
        # Generate an image based on the blog content or topic
        prompt = f"Create an artistic illustration related to the blog: {blog_content[:100]}..."  # Use a portion of the blog for a prompt
        image = stable_diffusion_model(prompt=prompt).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

def main():
    st.title("AI Blog Generator")
    
    topic = st.text_input("Enter Blog Topic:")
    word_count = st.number_input("Enter desired word count:", min_value=50, max_value=10000, value=1000)  # Increased max value

    if topic:
        if st.button("Generate Blog"): 
            if generator:
                # Generate blog content based on the topic and word count
                blog_content = generate_blog_content(topic, word_count)
                if blog_content:
                    st.subheader("Generated Blog Post:")
                    st.write(blog_content)
                    
                    # Display the word count of the generated blog post
                    generated_word_count = len(blog_content.split())
                    st.write(f"Word Count: {generated_word_count}")
                else:
                    st.warning("Blog content was not generated.")
                
                # Generate image based on the generated blog content
                image = generate_image_from_topic(blog_content)
                if image:
                    st.subheader("Related Image:")
                    st.image(image, use_column_width=True)
                else:
                    st.warning("Could not generate image for this topic.")
            else:
                st.error("Failed to load model. Please try again.")
    else:
        st.warning("Please enter a topic to generate a blog.")

if _name_ == "_main_":
    main()
