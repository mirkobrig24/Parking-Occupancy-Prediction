
# Questo bash lancia i file di STAR e ST-ResNet per la generazione di previsioni fissando la sgtrttura della rete
# cioè considero un solo ramo e non metto nelssun elemento esterno


# STAR
cd STAR
python main.py

# St-ResNet
cd ../ST-ResNet
python main.py

