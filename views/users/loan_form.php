<div class="container d-flex justify-content-end">
    <a href="index.php?page=user&action=dashboard" class="text-decoration-none text-primary mb-2 d-inline-block">
        &larr; Back to dashboard
    </a>
</div>

<?php require '../controllers/map.php'; ?>

<div class="container py-5">

    <?php if (isset($result)): ?>
    <div class="mt-3 mb-5 card border-0 shadow-sm p-4 text-center" style="background-color: #f5f5f5; border-radius: 12px;">
        <?php if ($result['prediction']): ?>
        <div class="py-3 px-4 rounded-pill d-inline-block w-100" style="background-color: #8cd38c; color: #1e5c22;">
            <h3 class="fw-bold m-0 tracking-wide" style="font-size: 1.5rem; letter-spacing: 0.5px;">
                RECOMMENDATION: APPROVE
            </h3>
        </div>
        <?php elseif (!$result['prediction']): ?>
            <div class="py-3 px-4 rounded-pill d-inline-block w-100" style="background-color: red; color: black;">
            <h3 class="fw-bold m-0 tracking-wide" style="font-size: 1.5rem; letter-spacing: 0.5px;">
                RECOMMENDATION: DENIED
            </h3>
            </div>
        <?php endif; ?>

        <?php if ($result['uncertain']): ?>
        <div class="mt-2 rounded-pill d-inline-block" style="background-color: #FFD700; color: black;">
            <h3 class="fw-bold m-0 fs-6 px-3 py-2 tracking-wide" style="font-size: 1.5rem; letter-spacing: 0.5px;">
            MODEL UNCERTAIN. THE SUBMISSION WILL BE REVIEWED MORE PRECISELY BY A HUMAN
            </h3>
        </div>
        <?php endif; ?>
        
        <div class="mt-3 text-secondary small fw-medium" style="letter-spacing: 0.2px;">
            Model Confidence: <span class="text-dark"><?=round(1.0/(1.0+exp(- (1/(float)$result['uncertainty_margin'])*abs((float)$result['decision_score'] - 1.1))),4)*100?>%</span> 
        </div>
    </div>

    <!-- <php if (!$result['prediction']): ?>
        <div class="card shadow-sm border-0 mb-4" style="background-color: #f8f9fa; border-radius: 15px;">
            <div class="card-body p-4">
                SHORT EXPLANATION : <br>
                FOLLOWING ARE THE MAIN REASON DETECTED BY OUR AI TO REJECT YOUR APPLICATION <br>
                <php foreach($result['explanation'] as $reason)?>
                FEATURE YOU MAY WANT TO IMPROVE : <= $reason['feature']?> <br>
                IMPACT IT HAD ON OUR AI CHOICES : <= $reason['impact']?> <br>
            </div>
        </div>
    <php endif; ?> -->
    <?php endif; ?>

    <div class="card shadow-sm border-0 mb-4" style="background-color: #f8f9fa; border-radius: 15px;">
        <div class="card-body p-4">
            <form action="index.php?page=run_model<?php if (isset($official_loan) && $official_loan) {echo "&action=official";}?>" method="POST">
                <div class="row g-3">
                    <div class="col-md">
                        <label class="form-label fw-bold small">Status of Existing Account</label>
                        <select class="form-select form-select-sm text-center" name="status_checking_account" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Status of existing checking account'] as $option): 
                            $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['status_checking_account']) && $fields['status_checking_account'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($old_name_to_modern_name[$option]); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Credit History</label>
                            <select class="form-select form-select-sm text-center" name="credit_history" required>
                                <option value="" selected disabled hidden>Select an option...</option>
                                <?php foreach($features_to_values['Credit history'] as $option): 
                                    $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['credit_history']) && $fields['credit_history'] == $optionId) ? 'selected' : '';
                            ?>
                                <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                                <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Savings account/bonds</label>
                            <select class="form-select form-select-sm text-center" name="savings_account" required>
                                <option value="" selected disabled hidden>Select an option...</option>
                                <?php foreach($features_to_values['Savings account/bonds'] as $option): 
                                    $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['savings_account']) && $fields['savings_account'] == $optionId) ? 'selected' : '';
                            ?>
                                <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($old_name_to_modern_name[$option]); ?></option>
                                <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Age</label>
                        <input value="<?= htmlspecialchars($fields['age'] ?? '')?>" type="number" name="age" id="age" min = "0" class="form-control text-center fw-bold" inputmode="integer" required>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Number of credits</label>
                        <input  value="<?= htmlspecialchars($fields['existing_credits'] ?? '')?>" type="number" name="existing_credits" id="existing_credits" min = "0" class="form-control text-center fw-bold" inputmode="integer" required>
                    </div>
                </div>

                <div class="row g-3 mt-2">
                    <div class="col-md-2">
                        <label class="form-label fw-bold small">Duration (month)</label>
                        <input value="<?= htmlspecialchars($fields['duration_months'] ?? '')?>" type="number" name="duration_months" id="duration_months" min = "0" class="form-control text-center fw-bold" inputmode="integer" required>
                    </div>
                    <div class="col-md-2">
                        <label class="form-label fw-bold small">Amount ($)</label>
                        <input value="<?= htmlspecialchars($fields['credit_amount'] ?? '')?>" type="number" name="credit_amount" id="credit_amount" min = "0" class="form-control text-center fw-bold" inputmode="integer" required>
                    </div>
                    <div class="col-md-2">
                        <label class="form-label fw-bold small">Installment rate in percentage of disposable income</label>
                        <input value="<?= htmlspecialchars($fields['installment_rate_pct'] ?? '')?>" type="number" name="installment_rate_pct" id="installment_rate_pct" min = "0" class="form-control text-center fw-bold" inputmode="integer" required>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Present employement since</label>
                        <select class="form-select form-select-sm text-center" name="employment_since" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Present employment since'] as $option): 
                                $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['employment_since']) && $fields['employment_since'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Job</label>
                        <select class="form-select form-select-sm text-center" name="job" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Job'] as $option): 
                                $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['job']) && $fields['job'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                </div>

                <div class="row g-3 mt-2">
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Purpose</label>
                        <select class="form-select form-select-sm text-center" name="purpose" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Purpose'] as $option): 
                                $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['purpose']) && $fields['purpose'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Other debtors/guarantors</label>
                        <select class="form-select form-select-sm text-center" name="other_debtors" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Other debtors / guarantors'] as $option): 
                                $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['other_debtors']) && $fields['other_debtors'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Present residence since</label>
                        <input value="<?= htmlspecialchars($fields['residence_since'] ?? '')?>" type="number" name="residence_since" id="residence_since" min = "0" class="form-control text-center fw-bold" inputmode="integer" required>
                    </div>
                </div>

                <div class="row g-3 mt-2">
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">STATUS & SEX</label>
                        <select class="form-select form-select-sm text-center" name="personal_status_sex" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Personal status and sex'] as $option): 
                                $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['personal_status_sex']) && $fields['personal_status_sex'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Property</label>
                        <select class="form-select form-select-sm text-center" name="property" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Property'] as $option): 
                                $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['property']) && $fields['property'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Other installement plans</label>
                            <select class="form-select form-select-sm text-center" name="other_installment_plans" required>
                                <option value="" selected disabled hidden>Select an option...</option>
                                <?php foreach($features_to_values['Other installment plans'] as $option): 
                                    $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['other_installment_plans']) && $fields['other_installment_plans'] == $optionId) ? 'selected' : '';
                            ?>
                                <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                                <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Housing</label>
                        <select class="form-select form-select-sm text-center" name="housing" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Housing'] as $option): 
                                $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['housing']) && $fields['housing'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Foreign worker</label>
                        <select class="form-select form-select-sm text-center" name="foreign_worker" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Foreign worker'] as $option): 
                                $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['foreign_worker']) && $fields['foreign_worker'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Telephone</label>
                        <select class="form-select form-select-sm text-center" name="telephone" required>
                            <option value="" selected disabled hidden>Select an option...</option>
                            <?php foreach($features_to_values['Telephone'] as $option): 
                                $optionId = $label_to_id[$option];
                            $isSelected = (isset($fields['telephone']) && $fields['telephone'] == $optionId) ? 'selected' : '';
                            ?>
                            <option value="<?= $optionId ?>" <?= $isSelected ?>><?= htmlspecialchars($option); ?></option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">	Number of people being liable to provide maintenance for</label>
                        <input value="<?= htmlspecialchars($fields['liable_dependents'] ?? '')?>" type="number" name="liable_dependents" id="liable_dependents" min = "0" class="form-control text-center fw-bold" inputmode="integer" required>
                    </div>
                </div>

                <div class="form-check mt-4">
                    <input class="form-check-input" type="checkbox" id="confirmData">
                    <label class="form-check-label small" for="confirmData">
                        I confirm that the uploaded data is accurate, complete, and formatted according to the required specifications.
                    </label>
                </div>

                <button type="submit" class="btn btn-dark w-100 py-3 mt-4 fw-bold fs-4 text-uppercase" style="letter-spacing: 2px;">
                    Run Model
                </button>
            </form>
        </div>
    </div>
</div>
