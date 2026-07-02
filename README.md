# ☁️ What Cloud Is It?

Um sistema de classificação de nuvens baseado em **MobileNetV3** para dispositivos móveis, capaz de identificar o tipo de formação nebulosa em tempo real a partir da câmera do smartphone — **sem depender de conexão com a internet**.

## Modelo

O backbone utilizado é o **MobileNetV3-Large**, escolhido por seu equilíbrio entre acurácia e baixo custo computacional, ideal para inferência em dispositivos móveis.

**Arquitetura da cabeça de classificação (customizada):**

1. Backbone MobileNetV3-Large (pesos pré-treinados no ImageNet)
2. Global Average Pooling (GAP)
3. Batch Normalization
4. Camada densa (Fully-Connected) — 128 neurônios, ativação ReLU
5. Batch Normalization
6. Dropout (taxa de 40%)
7. Camada de saída — ativação Softmax

## Como funciona o app

1. A câmera captura frames a cada 100 ms
2. Cada frame é recortado e redimensionado para 224×224 pixels
3. O frame é processado localmente pelo modelo TFLite
4. As últimas 10 predições são armazenadas em buffer
5. A classe mais frequente no buffer é exibida ao usuário, junto ao nível de confiança

<img width="200" alt="app_cumulus" src="https://github.com/user-attachments/assets/fe37aa89-1abd-406b-a8b7-875bbe012a8a" />
<img width="200" alt="app_clear_sky" src="https://github.com/user-attachments/assets/5f735407-8f78-466f-9a08-70254700137b" />
