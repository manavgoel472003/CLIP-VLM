# Project Title : VLM for Perception using Car Camera Feed (CLIP-VLM)

`This is the project milestone report for ISA project.`

## 1. What's Been done so far:
- Created a dataset of camera images and caption pairs using NuScenes and NuInteract. This includes for now, CAM_FRONT and CAM_BACK along with joint caption for both.
- Modified the Qwen VLM with a CLIP encoder so the model consumes fused multi-view features instead of a single built-in projector.
- For training the modified model, we freeze the main strcuture and add PEFT LoRA adapter to the model, so we don't have to do a full fine-tune.

## 2. Planned modules:
| Module | Status | Description |
| --- | --- | --- |
| Data pipeline (`data.py`) | Fully Functional | Loads the images to PIL for each sample and its corresponding joint caption from the manifest |
| Model (`model.py`) | Fully Functional | This consists of the modified architecture, the Fusion Connector takes the CLIP features and projects them to Qwen's hidden size, and then corss-attends to Qwen's tokens with a gated residual. |
| Training (`training.py`) | Partially Functional | Currently the model is being trained, on a small amount of epochs to produce similar captions, like those in nuInteract.  |
| Inference | Development Phase | Still figuriing out what sort of prompts to hard-code based on the trained model to get best results out of it |

## 3. Baseline module descriptions :
- **Model :**
  1. Each requested camera image (currently `CAM_FRONT` and `CAM_BACK`) is pushed through frozen OpenCLIP ViT-L/14. We keep one token per view, preserving directional cues like “front windshield” vs. “rear traffic.”
  2.  The stacked CLIP tokens are projected into Qwen’s hidden size, passed through shallow Transformer layers, and then attended to by Qwen’s visual prefix via a multi-head attention block with a learnable gate. The gate basically decides how strongly the fused vision overwrites Qwen’s frozen activations.
  3. Qwen’s backbone stays frozen, but LoRA adapters wrap the attention/MLP weights so a few million parameters can adapt. Combined with the connector, this is the only trainable path, keeping VRAM usage low.
  4. **Key path in code:**
     ```python
     q = self.teacher_tokens(images).to(self.cfg.device, dtype=target_dtype)
     e = self.ext_tokens(images).to(self.cfg.device, dtype=target_dtype)
     fused = self.connector(q, e)
     ```
     `teacher_tokens` captures Qwen’s frozen visual activations, `ext_tokens` holds CLIP features, and `self.connector` learns to align the two spaces before language generation.
- **Training :**
  1. For training we concatenate `[fused visual tokens, prompt embeddings, target caption embeddings]` and feed them into Qwen via `inputs_embeds`..
  3. Only the Fusion Connector weights and LoRA adapter matrices receive gradients; CLIP and the frozen Qwen backbone remain untouched.
  4. **How step training is done :**
     ```python
     # training.py 
     out = model.lm_step(images, prompt=prompt, labels_text=text)
     loss = out.loss          # cross-entropy over caption tokens only
     loss.backward()
     nn.utils.clip_grad_norm_(model.connector.parameters(), 1.0)
     opt.step()
     ```
     `lm_step` handles the concatenation/masking internally, and the trainer updates only the lightweight components after gradient clipping.


## 4. Challenges :
- The major challenge faced has been resources for fine-tuning the model as doing so requires high end A100 from google colab right now.
- The other issue is not having enough data, as using more data would require more resources. But, using frozen weights and fine-tuning can help counter some of that as we have pre-trained performance of some level.

## 5. What's left to do :
- Currently each epoch with on a 3000 sample dataset takes ~1.5 hrs, and we have only been able to train 5 epochs, which are yet to be tested.
- So, testing the current fine-tuned version against base model is the priority and then train for additional epochs to get hopefully better results.

## References
- nuScenes dataset & SDK:(https://www.nuscenes.org/).
- NuInteract dense captions: (public release at https://drive.google.com/drive/folders/1G7dlA0-8LZqYbO4k6G0bczwrt8sR6uH0).
- Qwen/Qwen2.5-VL-3B-Instruct: (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
- LoRA/PEFT adapters: (https://arxiv.org/abs/2106.09685).
