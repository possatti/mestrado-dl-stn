CIFAR-10 TODO
=============

## Tarefas
 - [x] Gerar imagens com distorções diferentes para cada época.
 - [ ] Verificar se as imagens passadas pela ST estão saindo pretas pq o dtype é float32.
 - [ ] Passar as imagens por diferentes épocas da STN e ver o que está acontendo.
 - [ ] Experimentar `LearningRateScheduler` para ver se consigo evitar que a STN desmanche.
 - [ ] Experimentar reduzir a resolução da imagem como output do ST.
 - [x] Cria opção `--cheap` para treinar com poucas imagens.
 - [x] Mudar nomes de "zuado" para "distorted".
 - [x] Usar rotações de -20 à +20.
 - [x] Pq treinar os dois modelos de uma vez está falhando? (Era por causa de `histogram_freq=1`. https://github.com/keras-team/keras/issues/4417)
