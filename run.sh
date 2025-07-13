#!/bin/bash

# Script de lancement du backend Lexica

echo "🚀 Démarrage du backend Lexica..."

# Vérifier si l'environnement virtuel existe
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python -m venv venv
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances
echo "📚 Installation des dépendances..."
pip install -r requirements.txt

# Vérifier le fichier .env
if [ ! -f ".env" ]; then
    echo "⚠️  Fichier .env manquant, copie de .env.example..."
    cp .env.example .env
    echo "📝 Veuillez configurer votre fichier .env avec vos clés API"
fi

# Lancer l'application

# Lancer l'application
echo "🎯 Lancement du serveur Flask..."
python app.py
