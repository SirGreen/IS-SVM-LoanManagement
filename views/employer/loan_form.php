<div class="container py-5">
    <div class="card shadow-sm border-0 mb-4" style="background-color: #f8f9fa; border-radius: 15px;">
        <div class="card-body p-4">
            <form action="index.php?page=run_model&action=official" method="POST">
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

    <div class="card shadow-sm border-0" style="background-color: #f1f1f1; border-radius: 15px;">
        <div class="card-body p-4">
            <div class="row align-items-center">
                <div class="col-md-7">
                    <div class="border border-primary border-dashed rounded d-flex align-items-center justify-content-center bg-white" 
                         style="height: 150px; border-style: dashed !important; border-width: 2px !important;">
                        <p class="mb-0 fw-bold">Drag & Drop your CSV file here</p>
                    </div>
                </div>
                <div class="col-md-5 d-flex flex-column gap-2 px-4">
                    <button class="btn btn-light text-muted border-0 py-2 disabled">Start</button>
                    <button class="btn btn-secondary py-2 fw-bold">Export PDF</button>
                </div>
            </div>
            
            <div class="form-check mt-3">
                <input class="form-check-input" type="checkbox" id="privacyPolicy">
                <label class="form-check-label small text-muted" for="privacyPolicy" style="font-size: 0.85rem;">
                    By processing this file, you confirm that all data complies with privacy regulations and has been handled according to our data protection policy.
                </label>
            </div>
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