# Correlation Matrix Editor

Application Streamlit pour:

- rechercher jusqu'a 30 securities via Yahoo Finance
- afficher des suggestions enrichies pendant la recherche
- ajouter des titres a une selection via un menu deroulant
- choisir une date precise de debut et une date precise de fin
- initialiser une matrice de correlation a partir des donnees de marche
- modifier manuellement la matrice
- conserver les correlations deja saisies quand la selection evolue
- forcer la symetrie et la diagonale a 1
- exporter la matrice finale en CSV

## Lancer le projet

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- La recherche utilise `yfinance.Search`.
- La matrice initiale est calculee depuis les rendements journaliers.
- L'app limite la selection a 30 titres.
