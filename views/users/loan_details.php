<?php
// Fonction pour gérer les couleurs des badges de statut
if (!function_exists('get_status_class')) {
    function get_status_class($status) {
        return match($status) {
            'Pending'  => 'bg-warning text-dark',
            'Approved' => 'bg-success text-white',
            'Refused'  => 'bg-danger text-white',
            default    => 'bg-secondary'
        };
    }
}
?>

<div class="container py-5" style="background-color: #f4f4f4; border-radius: 15px;">
    
    <!-- Bouton Retour -->
    <div class="row mb-4">
        <div class="col-12">
            <a href="index.php?page=user" class="btn btn-outline-secondary btn-sm rounded-pill px-3 text-decoration-none">
                ← Back to Dashboard
            </a>
        </div>
    </div>

    <!-- En-tête -->
    <div class="row mb-4">
        <div class="col-12">
            <span class="text-muted text-uppercase fw-bold small">Active Contracts</span>
            <h2 class="fw-bold text-dark">My Current Loans (<?= count($current_loan) ?>)</h2>
            <p class="text-muted">Click on a loan to view its full details and repayment progress.</p>
        </div>
    </div>

    <!-- Accordéon Bootstrap -->
    <div class="accordion" id="loansAccordion">
        <?php foreach ($current_loan as $index => $loan): 
            // Nettoyage de l'ID pour les attributs HTML (on retire le '#' pour éviter les bugs d'identifiants HTML)
            $html_id = ltrim($loan['application_id'], '#');
            
            $loan_amount = $loan['amount'];
            $remaining = $loan['remaining'];
            $monthly = $loan['monthly_payment'];
            $status = $loan['status'];
            
            // Calcul du pourcentage remboursé (si approuvé)
            $paid_amount = $loan_amount - $remaining;
            $progress_percent = $loan_amount > 0 ? round(($paid_amount / $loan_amount) * 100) : 0;
            
            // Le premier item reste ouvert par défaut
            $isOpen = ($index === 0);
        ?>
            
            <div class="card shadow-sm border-0 mb-3" style="border-radius: 12px; overflow: hidden;">
                
                <!-- En-tête de la ligne -->
                <div class="card-header bg-white border-0 p-4" id="heading<?= $html_id ?>">
                    <button class="btn w-100 text-start p-0 d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-3 collapse-button" 
                            type="button" 
                            data-bs-toggle="collapse" 
                            data-bs-target="#collapse<?= $html_id ?>" 
                            aria-expanded="<?= $isOpen ? 'true' : 'false' ?>" 
                            aria-controls="collapse<?= $html_id ?>">
                        
                        <div>
                            <span class="text-primary fw-bold small">ID: <?= $loan['application_id'] ?></span>
                            <h4 class="fw-bold mb-0 text-dark"><?= $loan['type'] ?></h4>
                        </div>

                        <div class="d-flex align-items-center gap-4 ms-md-auto me-md-3">
                            <div class="text-md-end">
                                <span class="text-muted d-block small">Total Amount</span>
                                <span class="fw-bold text-dark"><?= number_format($loan_amount, 2) ?> $</span>
                            </div>
                            <div class="text-md-end">
                                <span class="text-muted d-block small">Monthly Payment</span>
                                <span class="fw-bold text-primary"><?= number_format($monthly, 2) ?> $</span>
                            </div>
                            <span class="badge rounded-pill px-3 py-2 <?= get_status_class($status) ?>">
                                <?= $status ?>
                            </span>
                        </div>
                    </button>
                </div>

                <!-- Contenu déroulant -->
                <div id="collapse<?= $html_id ?>" 
                     class="accordion-collapse collapse <?= $isOpen ? 'show' : '' ?>" 
                     aria-labelledby="heading<?= $html_id ?>" 
                     data-bs-parent="#loansAccordion">
                    
                    <div class="card-body bg-light px-4 pb-4 pt-0">
                        <hr class="my-3 opacity-25">
                        
                        <!-- Section Progression (Masquée ou à 0% si le crédit est en attente) -->
                        <div class="row align-items-center mb-4">
                            <div class="col-md-3 mb-2 mb-md-0">
                                <span class="fw-bold text-muted small">Repayment Progress</span>
                            </div>
                            <div class="col-md-6 mb-2 mb-md-0">
                                <div class="progress" style="height: 10px; border-radius: 10px;">
                                    <div class="progress-bar <?= $status === 'Pending' ? 'bg-warning' : 'bg-success' ?>" 
                                         role="progressbar" 
                                         style="width: <?= $status === 'Pending' ? 0 : $progress_percent ?>%;" 
                                         aria-valuenow="<?= $progress_percent ?>" 
                                         aria-valuemin="0" 
                                         aria-valuemax="100">
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 text-md-end fw-bold <?= $status === 'Pending' ? 'text-warning' : 'text-success' ?> small">
                                <?php if ($status === 'Pending'): ?>
                                    Waiting for approval
                                <?php else: ?>
                                    <?= $progress_percent ?>% Paid (<?= number_format($paid_amount, 2) ?> $)
                                <?php endif; ?>
                            </div>
                        </div>

                        <!-- Grille d'informations -->
                        <div class="row g-4">
                            <!-- Caractéristiques financières -->
                            <div class="col-md-5">
                                <div class="bg-white p-3 rounded shadow-sm h-100">
                                    <h5 class="fw-bold mb-3 text-dark">Financial Overview</h5>
                                    
                                    <div class="d-flex justify-content-between mb-2 small border-bottom pb-1">
                                        <span class="text-muted">Remaining Balance</span>
                                        <span class="fw-bold text-danger"><?= number_format($remaining, 2) ?> $</span>
                                    </div>
                                    
                                    <div class="d-flex justify-content-between mb-2 small border-bottom pb-1">
                                        <span class="text-muted">Duration</span>
                                        <span class="fw-bold"><?= $loan['duration_months'] ?> Months</span>
                                    </div>
                                    
                                    <div class="d-flex justify-content-between mb-2 small border-bottom pb-1">
                                        <span class="text-muted">Interest Rate</span>
                                        <span class="fw-bold text-success"><?= $loan['interest_rate'] ?> %</span>
                                    </div>
                                    
                                    <div class="d-flex justify-content-between small">
                                        <span class="text-muted">Start Date</span>
                                        <span class="fw-bold"><?= $loan['start_date'] ?></span>
                                    </div>
                                </div>
                            </div>

                            <!-- Calendrier / Échéance à venir -->
                            <div class="col-md-7">
                                <div class="bg-white p-3 rounded shadow-sm h-100">
                                    <h5 class="fw-bold mb-3 text-dark">Status & Actions</h5>
                                    <?php if ($status === 'Pending'): ?>
                                        <div class="alert alert-warning border-0 small mb-0">
                                            <h6 class="fw-bold">Application under review</h6>
                                            Our teams (and our AI model) are currently analyzing your request. You will receive an email confirmation as soon as it is validated.
                                        </div>
                                    <?php else: ?>
                                        <table class="table table-sm table-borderless align-middle mb-0 small">
                                            <thead>
                                                <tr class="text-muted border-bottom">
                                                    <th>Next Payment</th>
                                                    <th>Type</th>
                                                    <th class="text-end">Amount</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr class="border-bottom">
                                                    <td class="py-2 fw-bold">Next Month Due</td>
                                                    <td>Principal + Interest</td>
                                                    <td class="text-end fw-bold text-primary"><?= number_format($monthly, 2) ?> $</td>
                                                </tr>
                                                <tr>
                                                    <td colspan="3" class="text-center text-muted pt-3" style="font-size: 0.8rem;">
                                                        To download your full official amortization plan for contract <strong><?= $loan['application_id'] ?></strong>, please head to your documents area.
                                                    </td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    <?php endif; ?>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>

        <?php endforeach; ?>
    </div>
</div>

<style>
/* Reset focus et styles natifs Bootstrap sur l'accordéon */
.collapse-button:focus {
    box-shadow: none;
}
.collapse-button:not(.collapsed) {
    color: inherit;
    background-color: transparent;
}
</style>