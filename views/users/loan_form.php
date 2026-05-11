<div class="container d-flex justify-content-end">
    <a href="index.php?page=user&action=dashboard" class="text-decoration-none text-primary mb-2 d-inline-block">
        &larr; Back to dashboard
    </a>
</div>

<div class="container py-5">
    <div class="card shadow-sm border-0 mb-4" style="background-color: #f8f9fa; border-radius: 15px;">
        <div class="card-body p-4">
            <form action="index.php?page=run_model<?php if (isset($official_loan) && $official_loan) {echo "&action=official";}?>" method="POST">
                <div class="row g-3">
                    <div class="col-md">
                        <label class="form-label fw-bold small">Existing Account ?</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Credit History</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Savings account</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Age</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md">
                        <label class="form-label fw-bold small">Number of credits</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                </div>

                <div class="row g-3 mt-2">
                    <div class="col-md-2">
                        <label class="form-label fw-bold small">Duration (month)</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md-2">
                        <label class="form-label fw-bold small">Amount ($)</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md-2">
                        <label class="form-label fw-bold small">Installment rate</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Present employement since</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small">Job</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                </div>

                <div class="row g-3 mt-2">
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Purpose</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Other debtors</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Present residence since</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                </div>

                <div class="row g-3 mt-2">
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">STATUS & SEX</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Property</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label fw-bold small">Other installement plan</label>
                        <input type="text" class="form-control text-center fw-bold" readonly>
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

<style>
    /* Styles personnalisés pour correspondre exactement au visuel */
    .form-control[readonly] {
        background-color: #ebebeb;
        border: 1px solid #ccc;
        color: #000;
    }
    .form-label {
        margin-bottom: 0.2rem;
    }
    .border-dashed {
        border-color: #0d6efd !important;
    }
</style>