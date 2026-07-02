# What Cloud Is It?
O "What Cloud Is It" é um aplciativo que faz a classificação de nuvens em 6 classes distintas por meio de uma rede neural construída com o backbone convolucional MobileNetV3-Large. O modelo foi treinado nos conjuntos GCD e CCSN para a classificação de nuvens.

O aplicativo faz capturas a cada 100ms, que são armazenadas em um buffer de predições. A classificação mais votada nas últimas 10 classifficações do buffer são apresentadas ao usuário como a classificação final.

<img width="200" alt="app_cumulus" src="https://github.com/user-attachments/assets/fe37aa89-1abd-406b-a8b7-875bbe012a8a" />
<img width="200" alt="app_clear_sky" src="https://github.com/user-attachments/assets/5f735407-8f78-466f-9a08-70254700137b" />
