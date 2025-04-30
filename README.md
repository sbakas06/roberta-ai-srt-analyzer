
# 🤖 Roberta AI – Analizzatore di file SRT

**Roberta AI** è una web app realizzata in Streamlit che analizza file `.srt` (sottotitoli) per rilevare automaticamente errori di trascrizione, grammaticali, ortografici o incongruenze logiche, grazie all'uso dell'IA OpenAI.

> 🧠 Sviluppato da Andrea Barilà per IDRA Srl.

---

## 🚀 Funzionalità

- Caricamento file `.srt`
- Analisi intelligente con modelli OpenAI (GPT-4)
- Rilevamento e correzione di snippet sospetti
- Output in tabella con motivazioni
- Storico delle analisi e possibilità di scaricare il report

---

## 🛠️ Requisiti

- Python 3.10+
- Account OpenAI con chiave API valida

---

## 🔧 Setup locale

1. **Clona il repository**

```bash
git clone https://github.com/TUO_USERNAME/roberta-ai-srt-analyzer.git
cd roberta-ai-srt-analyzer


2. **Crea un ambiente virtuale**

```bash
python3 -m venv venv
source venv/Scripts/activate #solo Windows


3. **Installa le dipendenze**

```bash
pip install -r requirements.txt


4. **Avvia l'app**

```bash
streamlit run nome_del_file.py



Tutti i diritti riservati © 2025 — Andrea Barilà per IDRA Srl.

