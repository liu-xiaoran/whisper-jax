# from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
# import jax.numpy as jnp

# from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
# import jax.numpy as jnp

# checkpoint_id = "sanchit-gandhi/whisper-small-hi"
# # convert PyTorch weights to Flax
# model = FlaxWhisperForConditionalGeneration.from_pretrained(checkpoint_id, from_pt=True)
# # push converted weights to the Hub
# model.push_to_hub(checkpoint_id)

# # now we can load the Flax weights directly as required
# pipeline = FlaxWhisperPipline(checkpoint_id, batch_size=16)




from whisper_jax import FlaxWhisperPipline

# # # instantiate pipeline
# # pipeline = FlaxWhisperPipline("openai/whisper-tiny")

# # instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-small")

# JIT compile the forward call - slow, but we only do once
text = pipeline("test.mp3")

print(text)

print("-------------------------------")

# used cached function thereafter - super fast!!
text2 = pipeline("test.mp3")

print(text2)