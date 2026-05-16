<div class="container py-5" style="background-color: #f4f4f4; border-radius: 15px;">
    
    <!-- Bouton Retour -->
    <div class="row mb-4">
        <div class="col-12">
            <a href="index.php?page=user" class="btn btn-outline-secondary btn-sm rounded-pill px-3 text-decoration-none">
                ← Back to Dashboard
            </a>
        </div>
    </div>

    <!-- En-tête Principal -->
    <div class="row mb-4">
        <div class="col-md-8">
            <span class="text-muted text-uppercase fw-bold small">AI Simulation Result</span>
            <h2 class="fw-bold text-dark">Simulation Details <span class="text-muted fs-4 fw-normal"><?= $sim_id ?></span></h2>
            <p class="text-muted">Simulated on <?= $date ?> • Powered by Credit Scoring Engine</p>
        </div>
        <div class="col-md-4 text-md-end d-flex align-items-center justify-content-md-end">
            <span class="badge rounded-pill px-4 py-2 fs-5 <?= $badge_class ?>">
                <?= $status ?>
            </span>
        </div>
    </div>

    <div class="row g-4">
        
        <!-- COLONNE GAUCHE : Score de l'IA & Décision -->
        <div class="col-lg-5">
            <div class="card h-100 shadow-sm border-0 p-4 text-center">
                <h4 class="fw-bold mb-3 text-start text-dark">AI Evaluation</h4>
                
                <div class="my-4">
                    <!-- Statut en gros iconisé -->
                    <?php if ($is_approved): ?>
                        <div class="display-1 text-success mb-2">✓</div>
                        <h3 class="fw-bold <?= $status_color ?>"><?= $status_title ?></h3>
                        <p class="text-muted px-3 small">Based on your simulation data, you meet our pre-eligibility criteria for this financial project.</p>
                    <?php else: ?>
                        <div class="display-1 text-danger mb-2">✕</div>
                        <h3 class="fw-bold <?= $status_color ?>"><?= $status_title ?></h3>
                        <p class="text-muted px-3 small">Unfortunately, current criteria do not allow us to pre-approve this simulation automatically.</p>
                    <?php endif; ?>
                </div>

                <hr class="opacity-25 my-4">

                <!-- Analyse du risque de crédit -->
                <div class="text-start">
                    <div class="d-flex justify-content-between mb-1">
                        <span class="fw-bold text-muted small">Estimated Default Risk Score</span>
                        <span class="badge <?= $risk_class ?> px-2 rounded"><?= $risk_text ?> (<?= $risk_score ?>%)</span>
                    </div>
                    <div class="progress" style="height: 12px; border-radius: 10px;">
                        <div class="progress-bar <?= $risk_class ?>" 
                             role="progressbar" 
                             style="width: <?= $risk_score ?>%;" 
                             aria-valuenow="<?= $risk_score ?>" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                    <small class="text-muted d-block mt-2 style-italic" style="font-size: 0.75rem;">
                        This score is calculated based on historical statistical patterns (similar to the Statlog credit engine).
                    </small>
                </div>
            </div>
        </div>

        <!-- COLONNE DROITE : Conditions Financières Simulées -->
        <div class="col-lg-7">
            <div class="card h-100 shadow-sm border-0 p-4 d-flex flex-column justify-content-between">
                <div>
                    <h4 class="fw-bold mb-4 text-dark">Simulated Financial Conditions</h4>
                    
                    <div class="row g-3">
                        <div class="col-sm-6">
                            <div class="bg-light p-3 rounded border-0">
                                <span class="text-muted d-block small">Requested Amount</span>
                                <span class="fs-4 fw-bold text-dark"><?= number_format($amount, 2) ?> $</span>
                            </div>
                        </div>
                        <div class="col-sm-6">
                            <div class="bg-light p-3 rounded border-0">
                                <span class="text-muted d-block small">Simulated APR (Rate)</span>
                                <span class="fs-4 fw-bold text-success"><?= $rate ?> %</span>
                            </div>
                        </div>
                        <div class="col-sm-6">
                            <div class="bg-light p-3 rounded border-0">
                                <span class="text-muted d-block small">Duration</span>
                                <span class="fs-4 fw-bold text-dark"><?= $duration ?> Months</span>
                            </div>
                        </div>
                    </div>

                    <!-- Note d'information -->
                    <div class="mt-4 p-3 rounded bg-white border" style="border-style: dashed !important;">
                        <h6 class="fw-bold text-dark mb-1 small">ℹ️ Important Disclaimer</h6>
                        <p class="text-muted mb-0 smal" style="font-size: 0.8rem;">
                            This simulation is strictly informative and does not constitute a contractual credit offer. Interest rates and access conditions may vary when filing an official request depending on supporting documents.
                        </p>
                    </div>
                </div>

                <!-- Call To Action dynamique tout en bas -->
                <div class="mt-4 pt-3 border-top text-end">
                    <?php if ($is_approved): ?>
                        <div class="d-flex flex-column flex-sm-row justify-content-between align-items-center gap-3">
                            <span class="text-muted text-start small">Satisfied with these conditions? Turn this simulation into a formal request.</span>
                            <a href="index.php?page=user&action=apply&from_sim=<?= $sim_id ?>" class="btn btn-primary rounded-pill px-4 fw-bold text-nowrap">
                                Apply Officially Now →
                            </a>
                        </div>
                    <?php else: ?>
                        <a href="index.php?page=user&action=simulation" class="btn btn-outline-primary rounded-pill px-4 fw-bold">
                            Try Another Simulation
                        </a>
                    <?php endif; ?>
                </div>

            </div>
        </div>

    </div>
</div>