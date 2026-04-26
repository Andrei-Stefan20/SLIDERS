# Note riunione, 23 aprile 2026

File Excalidraw nella cartella, da caricare su https://excalidraw.com/


## Task tecnici

- **Metriche di distanza.** Implementare la distanza di Manhattan nella pipeline di valutazione e confrontarla con cosine similarity ed euclidea.

- **Ablation expansion rate.** Testare il SAE con expansion rate 16x e 32x (oltre all'attuale 8x) per trovare il bilanciamento ideale tra sparsità e risoluzione delle feature.

- **SAE su singole patch invece che sul CLS.** Il flag `--use-patches` esiste già in `extract_embeddings.py` e in `DINOEncoder`, ma quando attivato produce tensori `(N, 256, 1024)` che rompono tutto quello che viene dopo (SAE training, build_index, UI si aspettano tutti `(N, 1024)`). Bisogna trattare ogni patch come un embedding indipendente, appiattire a `(N*256, 1024)` in estrazione, ripetere ogni path 256 volte in `image_paths.json` e deduplicare per path nel retrieval. SAE training e FAISS index non cambiano perché operano su vettori piatti come sempre, e ogni patch diventa un dato di training separato. Il SAE impara feature spazialmente localizzate anziché globali, e il retrieval può trovare immagini dove una piccola regione specifica è rilevante.

- **Impatto del testo.** Confrontare il retrieval su dataset con e senza caption per misurare quanto aiuta l'allineamento testo-immagine.

- **Sostituire CLIP alignment con Qwen come giudice.** Usiamo Qwen per generare i nomi e poi li valutiamo con CLIP, che è addestrato su caption brevi e codifica il linguaggio in modo diverso. L'alternativa è dare a Qwen un nome e le top-K immagini di una feature e chiedergli un punteggio di coerenza. Confrontare i risultati su un campione prima di decidere quale metrica tenere.

- **Validazione delle metriche di feature.**
  - Definire una metrica aggregata che combini purity, lift direzionale e allineamento semantico.
  - Stabilità al seed. Riaddestrare il SAE con seed diversi e misurare quanto variano monosemanticity e steering faithfulness. Se variano molto i risultati non sono riproducibili.
  - Bias del ranking. Le feature selezionate da `rank_diverse_mmr` potrebbero essere quelle più facili da nominare, non le più utili al retrieval. Confrontare recall@K con feature selezionate dal ranking vs feature selezionate a caso.
  - Valutazione contrastiva dei nomi in stile CE-Bench. Invece di misurare CLIP alignment, valutare se il nome di una feature permette di distinguere le immagini con alta attivazione (le top-N immagini dove quella feature si attiva di più) da quelle con bassa attivazione (le bottom-N dove non si attiva quasi). Il test è: dato solo il nome, un classificatore riesce a separare i due gruppi meglio del caso? Riferimento: [CE-Bench (arXiv:2509.00691)](https://arxiv.org/abs/2509.00691).

---

## Ricerca

- **VLM as a judge.** Usare un VLM per valutare in modo automatico se le top-activating images di una feature SAE rappresentano un concetto coerente.

- **DINO vs CLIP.** Confrontare le performance di retrieval del nostro approccio visivo (DINOv2) contro modelli allineati al testo (CLIP) su dataset etichettati.

- **Benchmarking su LAION.** Usare LAION come dataset open-source per confrontare modelli con e senza caption su uno stesso benchmark.

- **SAE gerarchico.** Esplorare architetture a più livelli dove ogni livello opera sulle attivazioni di quello precedente per estrarre concetti sempre più fini. Non è chiaro quanti livelli usare, come addestrarli, come esporli nella UI, o se la gerarchia che viene fuori è davvero interpretabile.

---

## Sperimentazioni

- **Dataset rari.** Valutare il SAE su dataset con concetti visivi rari per capire se il modello generalizza, e su dataset molto semplici per vedere l'altro estremo.

- **Composed image retrieval.** SLIDER fa già questo in pratica, ma non lo abbiamo mai misurato su niente di formale. Esistono dataset con triple (immagine, modifica, target) già annotate tipo CIRR o FashionIQ, vale la pena girare la pipeline su uno di quelli e vedere dove si posiziona rispetto ad altri approcci.

- **Fine-tuning di DINOv2.** Testare se fare fine-tuning dell'encoder (invece di usarlo out-of-the-box) produce feature più interpretabili e più facili da nominare.

- **Miglioramento del naming.**
  - Testare VLM più grandi (`Qwen3-VL-7B`, `Qwen3-VL-32B`) e confrontare la qualità dei nomi (CLIP alignment, cross-model score) per capire se il guadagno giustifica il costo.
  - Più crop e crop più grandi (`--n-crops 16 --crop-size 128`) per dare più contesto al VLM. Da misurare se c'è davvero un guadagno di qualità.
  - Naming iterativo. Generare un nome, poi chiedere al VLM se quel nome si applica anche alle immagini a bassa attivazione. Se risponde sì il nome è troppo generico e va raffinato. Ripetere fino a convergenza o per max N round.

---

## UI

- **Mappa di attivazione per immagine.** Aggiungere una modal per ogni immagine risultato che mostri una griglia 16x16 sovrapposta, dove ogni patch è colorata in base all'intensità di attivazione della feature dello slider attivo (o della feature con attivazione più alta se nessuno slider è attivo). L'overlay deve mostrare il nome della feature associata a quella patch.

---

## Problemi dell'approccio

- **Lo steering non è causale.** Aggiungere una decoder column alla query sposta il vettore verso la zona dello spazio DINO dove stanno le immagini con quella feature, ma quella zona contiene anche immagini con altre proprietà che si trovano lì per motivi diversi. In letteratura ([arXiv:2509.00749, 2025](https://arxiv.org/pdf/2509.00749)) risulta che solo il 10-15% delle feature SAE sui vision model producono uno shift di retrieval causale. Il reranking per attivazione aiuta ma non risolve il problema.

- **ReLU+L1 vs TopK.** Usiamo ReLU+L1 di default ma potrebbe valere la pena provare TopK. Il problema di L1 è che spinge tutte le attivazioni verso zero durante il training, quindi il SAE tende ad imparare valori più bassi del necessario. TopK evita questo perché tiene solo le K attivazioni più alte e azzera le altre, senza toccare i valori. Da valutare se cambiare il default.

- **Neuroni morti e come riattivarli.** Durante il training alcuni neuroni smettono di attivarsi su qualsiasi input e restano a zero per sempre. Il codice li rileva e li reinizializza di colpo, ma questo causa un picco nella loss ogni volta perché il neurone cambia comportamento improvvisamente. Un approccio migliore è dare ai neuroni morti un segnale di gradiente piccolo e continuo, senza toccare i pesi, così si riprendono gradualmente senza destabilizzare il training (non implementato).

- **Il CLS token non vede i dettagli locali.** Il FAISS index è costruito sui CLS token, che riassumono l'immagine intera. Una feature che risponde a una lesione piccola su una foglia pesa pochissimo nel CLS perché coinvolge magari 3-4 patch su 256. Due immagini, una con e una senza quella lesione, possono avere CLS quasi identici, quindi steerare su quella feature non cambia quasi nulla nel retrieval. L'unica soluzione sarebbe un indice sui patch token, o selezione di patch.

- **Le metriche di valutazione non misurano quello che conta.** CLIP alignment alta non significa che il nome sia buono. "Verde" su un dataset di piante ha alignment altissima ma non dice niente. La purity/monosemanticity dipende da quanto sono fini le classi nel dataset. Il test che conta davvero è se il nome permette di separare immagini HIGH da LOW, e non lo facciamo.
