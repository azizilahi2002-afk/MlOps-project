# Dockerfile
# Utiliser une image Python officielle et légère
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances et les installer
# On utilise requirements-dev pour inclure pytest, black, etc. pour d'éventuels tests dans le CI
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copier le code source de l'application
COPY src/ ./src/
COPY models/ ./models/

# Créer un utilisateur non-root pour des raisons de sécurité
RUN useradd --create-home --shell /bin/bash app
USER app

# Exposer le port sur lequel l'API va écouter
EXPOSE 8000

# Commande pour lancer l'API avec uvicorn
# --host 0.0.0.0 est important pour que l'API soit accessible depuis l'extérieur du conteneur
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
