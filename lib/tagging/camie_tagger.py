from camie_tagger import CamieTagger

model = CamieTagger('path/to/model_weights')
tags = model.tag_image('path/to/image.png')