/**
 * Script principal de l'application
 * Gère l'affichage des graphiques et les interactions UI
 */

document.addEventListener('DOMContentLoaded', function() {
    
    // 1. INITIALISATION DES GRAPHIQUES (Pour le Data Scientist & Dashboard)
    const modelCtx = document.getElementById('modelChart');
    if (modelCtx) {
        new Chart(modelCtx, {
            type: 'line',
            data: {
                labels: ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'],
                datasets: [{
                    label: 'Précision du modèle (%)',
                    data: [94, 95, 94.8, 96, 97.2, 97.5, 98.2],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { beginAtZero: false, min: 90 }
                }
            }
        });
    }

    // 2. GESTION DE LA SIDEBAR (Actif/Inactif)
    const currentPath = window.location.search;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (currentPath.includes(link.getAttribute('href'))) {
            link.classList.add('active');
        }
    });

    // 3. PRÉPARATION DES DONNÉES DE FORMULAIRE
    // Avant la soumission PHP classique, on peut nettoyer ou valider les données
    const mainForms = document.querySelectorAll('form');
    mainForms.forEach(form => {
        form.addEventListener('submit', function() {
            const btn = form.querySelector('button[type="submit"]');
            if (btn) {
                // Feedback visuel pendant que PHP traite et envoie à l'API FastAPI
                btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Envoi...';
                btn.classList.add('disabled');
            }
        });
    });

    // 4. SYSTÈME DE NOTIFICATION (Toasts Bootstrap)
    // Utile pour afficher un succès après un retour d'API traité par PHP
    const showToast = (message, type = 'success') => {
        const toastContainer = document.getElementById('toast-container');
        if (!toastContainer) return;

        const id = Date.now();
        const toastHTML = `
            <div id="toast-${id}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">${message}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>`;
        
        toastContainer.innerHTML += toastHTML;
        const toastElement = document.getElementById(`toast-${id}`);
        const bsToast = new bootstrap.Toast(toastElement);
        bsToast.show();
    };
});

/*********************************/ 
/************* MODAL *************/ 
/*********************************/ 


// // Handle Edit Buttons
// document.querySelectorAll('.review-btn').forEach(function(btn) {
//     btn.addEventListener('click', function() {
//         var id = this.getAttribute('data-review-id');
//         var name = this.getAttribute('data-edit-name');
//         var table = this.getAttribute('data-edit-table');
        
//         document.getElementById('modal-title').innerText = 'Edit Entry';
//         document.getElementById('add-entry-form').action = "<?= $baseUrl ?>/admin/updateRef";
//         document.getElementById('entry-id').value = id;
//         document.getElementById('entry-table').value = table;
//         document.getElementById('entry-name').value = name;
        
//         // Show modal (assuming script.js handles the data-modal click, but we manually trigger it if needed)
//         document.getElementById('add-modal').classList.add('active');
//     });
// });

//   // Handle modal close
//   document.querySelectorAll('.modal-close').forEach(function(btn) {
//       btn.addEventListener('click', function() {
//           document.getElementById('add-modal').classList.remove('active');
//       });
//   });