// =============================================
//  script.js — KinVerif.ai
//  Contient : ADN animé + upload + résultats + métriques
// =============================================


// -----------------------------------------------
//  1. ANIMATION ADN EN ARRIÈRE-PLAN (Canvas)
// -----------------------------------------------
(function () {
  const canvas = document.getElementById('dnaCanvas');
  const ctx    = canvas.getContext('2d');

  // On définit plusieurs "brins" d'ADN à des positions différentes
  const strands = [
    { x: 0.08, speed: 0.6,  phase: 0,    amplitude: 38, color: '#9D8FE0' },
    { x: 0.30, speed: 0.45, phase: 1.5,  amplitude: 32, color: '#BDB0EC' },
    { x: 0.55, speed: 0.55, phase: 0.8,  amplitude: 36, color: '#7C6FCD' },
    { x: 0.78, speed: 0.5,  phase: 2.2,  amplitude: 34, color: '#CEC8F4' },
    { x: 0.95, speed: 0.65, phase: 1.0,  amplitude: 30, color: '#9D8FE0' },
  ];

  let t = 0; // temps qui avance à chaque frame

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function dessinerBrin(strand) {
    const xCenter = strand.x * canvas.width;
    const amp     = strand.amplitude;
    const spacing = 28;  // espacement vertical entre les "marches" d'ADN
    const hauteur = canvas.height;

    // Couleur avec transparence
    ctx.strokeStyle = strand.color;
    ctx.lineWidth   = 1.6;

    // --- Brin gauche (sinusoïde) ---
    ctx.beginPath();
    for (let y = 0; y < hauteur; y += 4) {
      const angle = (y / 60) + t * strand.speed + strand.phase;
      const x     = xCenter + Math.sin(angle) * amp;
      if (y === 0) ctx.moveTo(x, y);
      else         ctx.lineTo(x, y);
    }
    ctx.globalAlpha = 0.5;
    ctx.stroke();

    // --- Brin droit (sinusoïde décalée de PI) ---
    ctx.beginPath();
    for (let y = 0; y < hauteur; y += 4) {
      const angle = (y / 60) + t * strand.speed + strand.phase + Math.PI;
      const x     = xCenter + Math.sin(angle) * amp;
      if (y === 0) ctx.moveTo(x, y);
      else         ctx.lineTo(x, y);
    }
    ctx.globalAlpha = 0.5;
    ctx.stroke();

    // --- Barreaux horizontaux ("bases nucléiques") ---
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = 0.25;
    for (let y = 0; y < hauteur; y += spacing) {
      const angle   = (y / 60) + t * strand.speed + strand.phase;
      const xGauche = xCenter + Math.sin(angle) * amp;
      const xDroit  = xCenter + Math.sin(angle + Math.PI) * amp;

      // Couleur alternée pour les bases
      const paire = Math.floor(y / spacing) % 4;
      if (paire === 0)      ctx.strokeStyle = '#AFA9EC';
      else if (paire === 1) ctx.strokeStyle = '#C4BCED';
      else if (paire === 2) ctx.strokeStyle = '#9D8FE0';
      else                  ctx.strokeStyle = '#D4CFEF';

      ctx.beginPath();
      ctx.moveTo(xGauche, y);
      ctx.lineTo(xDroit, y);
      ctx.stroke();

      // Petits cercles aux extrémités des barreaux
      ctx.fillStyle = ctx.strokeStyle;
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      ctx.arc(xGauche, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(xDroit, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;
  }

  function animer() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    t += 0.008; // vitesse de défilement (plus petit = plus lent)

    strands.forEach(s => dessinerBrin(s));

    requestAnimationFrame(animer);
  }

  resize();
  window.addEventListener('resize', resize);
  animer();
})();


// -----------------------------------------------
//  2. GESTION DES PHOTOS
// -----------------------------------------------
let photo1Chargee = false;
let photo2Chargee = false;

function chargerImage(input, idZone, idPreview, idNom) {
  const fichier = input.files[0];
  if (!fichier) return;

  const lecteur = new FileReader();
  lecteur.onload = function (e) {
    document.getElementById(idPreview).src = e.target.result;

    const nom = fichier.name.length > 22
      ? fichier.name.substring(0, 19) + '...'
      : fichier.name;
    document.getElementById(idNom).textContent = nom;

    if (idZone === 'zone1') {
      document.getElementById('vide1').style.display  = 'none';
      document.getElementById('pleine1').style.display = 'flex';
      photo1Chargee = true;
    } else {
      document.getElementById('vide2').style.display  = 'none';
      document.getElementById('pleine2').style.display = 'flex';
      photo2Chargee = true;
    }

    document.getElementById(idZone).classList.add('active');
    mettreAJourInterface();
  };
  lecteur.readAsDataURL(fichier);
}

function retirerImage(idZone, idPleine, idVide, idPreview, idInput) {
  document.getElementById(idPreview).src   = '';
  document.getElementById(idInput).value   = '';
  document.getElementById(idPleine).style.display = 'none';
  document.getElementById(idVide).style.display   = 'flex';
  document.getElementById(idZone).classList.remove('active');

  if (idZone === 'zone1') photo1Chargee = false;
  else                    photo2Chargee = false;

  document.getElementById('resultat').style.display = 'none';
  mettreAJourInterface();
}

function mettreAJourInterface() {
  const bouton = document.getElementById('btnAnalyser');
  const aide   = document.getElementById('aide');
  const etape  = document.getElementById('etapeIndicateur');

  if (photo1Chargee && photo2Chargee) {
    bouton.disabled    = false;
    aide.textContent   = 'Les deux photos sont prêtes. Lancez l\'analyse !';
    aide.style.color   = 'var(--succes)';
    etape.textContent  = 'Étape 2 / 2 — Prêt à analyser';
  } else if (photo1Chargee || photo2Chargee) {
    bouton.disabled    = true;
    aide.textContent   = 'Chargez la deuxième photo pour continuer.';
    aide.style.color   = 'var(--texte-3)';
    etape.textContent  = 'Étape 1 / 2 — Chargez vos photos';
  } else {
    bouton.disabled    = true;
    aide.textContent   = 'Chargez les deux photos pour activer l\'analyse.';
    aide.style.color   = 'var(--texte-3)';
    etape.textContent  = 'Étape 1 / 2 — Chargez vos photos';
  }
}


// -----------------------------------------------
//  3. ANALYSE (simulation — à remplacer par l'API)
// -----------------------------------------------
function lancerAnalyse() {
  const bouton   = document.getElementById('btnAnalyser');
  const spinner  = document.getElementById('spinner');
  const btnTexte = document.getElementById('btnTexte');
  const btnIco   = document.getElementById('btnIco');

  bouton.disabled        = true;
  spinner.style.display  = 'block';
  btnIco.style.display   = 'none';
  btnTexte.textContent   = 'Analyse en cours...';
  document.getElementById('resultat').style.display = 'none';

  // Simulation d'un appel réseau de 2 secondes
  // ── À REMPLACER PLUS TARD PAR : fetch('http://localhost:8000/verify', ...) ──
  setTimeout(function () {
    const score        = Math.floor(55 + Math.random() * 42);
    const estApparente = score >= 72;

    afficherResultat(score, estApparente);

    bouton.disabled       = false;
    spinner.style.display = 'none';
    btnIco.style.display  = 'block';
    btnTexte.textContent  = 'Lancer l\'analyse';
  }, 2200);
}


// -----------------------------------------------
//  4. AFFICHAGE DU RÉSULTAT
// -----------------------------------------------
function afficherResultat(score, estApparente) {
  const zone   = document.getElementById('resultat');
  const titre  = document.getElementById('resultatTitre');
  const badge  = document.getElementById('resultatBadge');
  const num    = document.getElementById('scoreChiffre');
  const barre  = document.getElementById('barreRemplie');
  const note   = document.getElementById('resultatNote');
  const icon   = document.getElementById('resIcon');

  if (estApparente) {
    zone.className    = 'resultat apparente';
    titre.textContent = 'Lien de parenté probable';
    titre.style.color = 'var(--succes)';
    badge.textContent = 'Apparentés';
    badge.className   = 'res-badge badge-succes';
    num.style.color   = 'var(--succes)';
    barre.className   = 'barre-fill succes';
    icon.className    = 'res-icon succes';
    icon.textContent  = '✓';
    icon.style.color  = 'var(--succes)';
    icon.style.fontWeight = '700';
    note.textContent  = 'Le modèle a détecté des similarités faciales significatives entre les deux personnes, '
                      + 'suggérant l\'existence d\'un lien de parenté. Ce résultat est indicatif et basé '
                      + 'sur l\'analyse des traits biométriques du visage.';
  } else {
    zone.className    = 'resultat non-apparente';
    titre.textContent = 'Aucun lien de parenté détecté';
    titre.style.color = 'var(--danger)';
    badge.textContent = 'Non apparentés';
    badge.className   = 'res-badge badge-danger';
    num.style.color   = 'var(--danger)';
    barre.className   = 'barre-fill danger';
    icon.className    = 'res-icon danger';
    icon.textContent  = '✕';
    icon.style.color  = 'var(--danger)';
    icon.style.fontWeight = '700';
    note.textContent  = 'Le modèle n\'a pas détecté de similarités faciales suffisantes. '
                      + 'Aucun lien de parenté n\'est suggéré. Ce résultat est indicatif et basé '
                      + 'sur l\'analyse des traits biométriques du visage.';
  }

  num.textContent        = score + '%';
  barre.style.width      = '0%';
  zone.style.display     = 'block';

  setTimeout(() => { barre.style.width = score + '%'; }, 60);
  zone.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}


// -----------------------------------------------
//  5. RÉINITIALISER
// -----------------------------------------------
function reinitialiser() {
  retirerImage('zone1', 'pleine1', 'vide1', 'preview1', 'input1');
  retirerImage('zone2', 'pleine2', 'vide2', 'preview2', 'input2');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}


// -----------------------------------------------
//  6. ANIMER LES CERCLES DE PRÉCISION (Accuracy)
//     Ces valeurs sont à mettre à jour quand vous
//     avez les vrais résultats de votre modèle.
// -----------------------------------------------
const metriques = {
  accuracy:  0.87,   // ← remplacez par votre vraie valeur, ex: 0.91 = 91%
  f1:        0.84,   // ← F1-Score de votre modèle
  precision: 0.89,   // ← Précision de votre modèle
};

function animerRing(idRing, idVal, valeur, delai) {
  const circonference = 201; // 2 * PI * 32 (rayon du cercle SVG)
  const offset = circonference * (1 - valeur);

  setTimeout(function () {
    document.getElementById(idRing).style.strokeDashoffset = offset;
    document.getElementById(idVal).textContent = Math.round(valeur * 100) + '%';
  }, delai);
}

// On lance l'animation quand la section est visible
// (IntersectionObserver = déclenche quand on fait défiler jusqu'à la section)
const accSection = document.querySelector('.section-accuracy');
if (accSection) {
  const observer = new IntersectionObserver(function (entries) {
    if (entries[0].isIntersecting) {
      animerRing('ringPrec', 'precVal', metriques.precision, 700);
      observer.disconnect(); // on n'anime qu'une seule fois
    }
  }, { threshold: 0.3 });
  observer.observe(accSection);
}