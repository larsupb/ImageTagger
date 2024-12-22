import gradio as gr

import config
from lib.upscaling import Upscalers


def tab_settings(state: gr.State):
    with gr.Tab("Settings"):
        def update_sbert_taggers(state_dict, taggers):
            config.update(state_dict, "sbert_taggers", taggers)

        def update_sbert_threshold(state_dict, value):
            config.update(state_dict, "sbert_threshold", value)

        def update_upscaler(state_dict, upscaler):
            config.update(state_dict, "upscaler", upscaler)

        def update_tagger_instruction(state_dict, instruction):
            config.update(state_dict, "tagger_instruction", instruction)


        with gr.Row("Captioning"):
            checkbox_sbert_taggers = gr.CheckboxGroup(choices=['joytag', 'wd14', 'florence'],
                                                      label="SBERT taggers", value=config.sbert_taggers(state.value))
            checkbox_sbert_taggers.change(update_sbert_taggers, inputs=[state, checkbox_sbert_taggers])

            slider_sbert_threshold = gr.Slider(value=config.sbert_threshold(state.value), minimum=0., maximum=1.0,
                                               label="SBERT threshold", step=0.01)
            slider_sbert_threshold.change(update_sbert_threshold, inputs=[state, slider_sbert_threshold])

            textbox_tagger_instruction = gr.Textbox(label="Tagger instruction", placeholder="Enter tagger instruction here..", lines=3,
                                                 interactive=True, value=config.tagger_instruction(state.value))
            textbox_tagger_instruction.change(update_tagger_instruction, inputs=[state, textbox_tagger_instruction])
        with gr.Row("Upscalers"):
            radio_upscalers = gr.Radio(choices=[upscaler.name for upscaler in Upscalers],
                                          label="Upscaler", value=config.upscaler(state.value))
            radio_upscalers.change(update_upscaler, inputs=[state, radio_upscalers])
