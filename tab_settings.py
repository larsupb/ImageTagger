import gradio as gr

import config
from lib.upscaling import Upscalers


def tab_settings(state: gr.State):
    with gr.Tab("Settings"):
        def update_combo_taggers(state_dict, taggers):
            config.update(state_dict, "combo_taggers", taggers)

        def update_florence_prompt(state_dict, prompt):
            florence_settings = state_dict["florence_settings"]
            florence_settings["prompt"] = prompt
            config.update(state_dict, "florence_settings", florence_settings)

        def update_upscaler(state_dict, upscaler):
            config.update(state_dict, "upscaler", upscaler)

        def update_tagger_instruction(state_dict, instruction):
            config.update(state_dict, "tagger_instruction", instruction)

        def update_rembg_model(state_dict, instruction):
            config.update(state_dict, "rembg", instruction)


        with gr.Row("Combo taggers"):
            checkbox_combo_taggers = gr.CheckboxGroup(choices=['florence', 'joytag', 'wd14'],
                                                      label="Combo taggers", value=config.combo_taggers(state.value))
            checkbox_combo_taggers.change(update_combo_taggers, inputs=[state, checkbox_combo_taggers])

        with gr.Row("Masking"):
            rembg_model = gr.Dropdown(label="Rembg model",
                                                  choices=["u2net_human_seg", "u2net", "u2net_cloth_seg"],
                                                  value=config.rembg(state.value)["model"])
            rembg_model.change(update_rembg_model, inputs=[state, rembg_model])

            textbox_tagger_instruction = gr.Textbox(label="Tagger instruction", placeholder="Enter tagger instruction here..", lines=3,
                                                 interactive=True, value=config.tagger_instruction(state.value))
            textbox_tagger_instruction.change(update_tagger_instruction, inputs=[state, textbox_tagger_instruction])
        with gr.Row("Florence-2"):
            textbox_florence_prompt = gr.Dropdown(label="Florence prompt",
                                                  choices=["<GENERATE_PROMPT>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
                                                  value=config.florence_settings(state.value)["prompt"])
            textbox_florence_prompt.change(update_florence_prompt, inputs=[state, textbox_florence_prompt])

        with gr.Group():
            gr.Markdown("### OpenAI-compatible VLM")

            def update_openai_setting(state_dict, key, value):
                openai_settings = config.openai_settings(state_dict)
                openai_settings[key] = value
                config.update(state_dict, "openai_settings", openai_settings)

            with gr.Row():
                openai_api_key = gr.Textbox(
                    label="API Key",
                    placeholder="Enter API key (leave empty for local Ollama)",
                    value=config.openai_settings(state.value)["api_key"],
                    type="password"
                )
                openai_api_key.change(
                    lambda s, v: update_openai_setting(s, "api_key", v),
                    inputs=[state, openai_api_key]
                )

                openai_base_url = gr.Textbox(
                    label="Base URL",
                    placeholder="http://localhost:11434/v1",
                    value=config.openai_settings(state.value)["base_url"]
                )
                openai_base_url.change(
                    lambda s, v: update_openai_setting(s, "base_url", v),
                    inputs=[state, openai_base_url]
                )

                openai_model = gr.Textbox(
                    label="Model",
                    placeholder="qwen3:32b",
                    value=config.openai_settings(state.value)["model"]
                )
                openai_model.change(
                    lambda s, v: update_openai_setting(s, "model", v),
                    inputs=[state, openai_model]
                )

            openai_prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe this image in detail.",
                value=config.openai_settings(state.value)["prompt"],
                lines=4
            )
            openai_prompt.change(
                lambda s, v: update_openai_setting(s, "prompt", v),
                inputs=[state, openai_prompt]
            )

        with gr.Row("Upscalers"):
            radio_upscalers = gr.Radio(choices=[upscaler.name for upscaler in Upscalers],
                                          label="Upscaler", value=config.upscaler(state.value))
            radio_upscalers.change(update_upscaler, inputs=[state, radio_upscalers])

            slider_upscale_target_megapixels = gr.Slider(value=config.upscale_target_megapixels(state.value),
                                                         minimum=0.1, maximum=10.0, label="Target megapixels",
                                                         step=0.1)
            slider_upscale_target_megapixels.change(
                lambda state_dict, value: config.update(state_dict, "upscale_target_megapixels", value),
                inputs=[state, slider_upscale_target_megapixels]
            )
