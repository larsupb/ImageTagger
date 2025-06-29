import gradio as gr

import config
from lib.upscaling import Upscalers


def tab_settings(state: gr.State):
    with gr.Tab("Settings"):
        def update_combo_taggers(state_dict, taggers):
            config.update(state_dict, "combo_taggers", taggers)

        def update_sbert_taggers(state_dict, taggers):
            config.update(state_dict, "sbert_taggers", taggers)

        def update_sbert_threshold(state_dict, value):
            config.update(state_dict, "sbert_threshold", value)

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

        with gr.Row("SBERT"):
            checkbox_sbert_taggers = gr.CheckboxGroup(choices=['joytag', 'wd14', 'florence'],
                                                      label="SBERT taggers", value=config.sbert_taggers(state.value))
            checkbox_sbert_taggers.change(update_sbert_taggers, inputs=[state, checkbox_sbert_taggers])

            slider_sbert_threshold = gr.Slider(value=config.sbert_threshold(state.value), minimum=0., maximum=1.0,
                                               label="SBERT threshold", step=0.01)
            slider_sbert_threshold.change(update_sbert_threshold, inputs=[state, slider_sbert_threshold])

            textbox_tagger_instruction = gr.Textbox(label="Tagger instruction", placeholder="Enter tagger instruction here..", lines=3,
                                                 interactive=True, value=config.tagger_instruction(state.value))
            textbox_tagger_instruction.change(update_tagger_instruction, inputs=[state, textbox_tagger_instruction])
        with gr.Row("Florence-2"):
            textbox_florence_prompt = gr.Dropdown(label="Florence prompt",
                                                  choices=["<GENERATE_PROMPT>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
                                                  value=config.florence_settings(state.value)["prompt"])
            textbox_florence_prompt.change(update_florence_prompt, inputs=[state, textbox_florence_prompt])

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
