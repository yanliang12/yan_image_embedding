import os
import yan_image_embedding

os.system('wget https://m.eyeofriyadh.com/news_images/2020/01/1f75d29d39631.jpg')

vector = yan_image_embedding.image_to_vector('1f75d29d39631.jpg')
