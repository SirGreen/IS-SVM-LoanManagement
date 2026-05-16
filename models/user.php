<?php
class User {
    public function __construct() {
    }

    public function get_request_history($user_id) {
        if ($_SESSION['role']==='user') {
            # We have only one user so we hard code the data. With multiple user (and a database)
            # we would have done a sql request to retrieve all the necessary data.
            $history = [
                [
                    "application_id" => "#AI82741",
                    "type"           => "Official Loan",
                    "amount"         => 2000,
                    "date"           => "05/04/2026",
                    "status"          => "Pending"
                ],
                [
                    "application_id" => "#BQ2834",
                    "type"           => "Simulation",
                    "amount"         => 2000,
                    "date"           => "04/04/2026",
                    "status"          => "Approved"
                ],
                [
                    "application_id" => "#BIA1224",
                    "type"           => "Simulation",
                    "amount"         => 4000,
                    "date"           => "04/04/2026",
                    "status"          => "Refused"
                ],
                [
                    "application_id" => "#BH1048",
                    "type"           => "Official Loan",
                    "amount"         => 50000,
                    "date"           => "23/06/2025",
                    "status"          => "Approved"
                ]
            ];
            return $history;
        }
    }

    public function get_loan_number($user_id) {
        if ($_SESSION['role']==='user') {
            # We have only one user so we hard code the data. With multiple user (and a database)
            # we would have done a sql request to retrieve all the necessary data.
            return 2;
        }
    }

    public function get_loan_pending(){
        if ($_SESSION['role']==='employer') {
            # Use sql request if database
            $history = [
                [
                    "application_id" => "#AI84841",
                    "amount"         => 2000,
                    "date"           => "05/04/2026",
                    "status"          => "Pending Review"
                ],
                [
                    "application_id" => "#AI82121",
                    "amount"         => 2000,
                    "date"           => "04/04/2026",
                    "status"          => "Pending Review"
                ],
                [
                    "application_id" => "#BIA1224",
                    "amount"         => 4000,
                    "date"           => "04/04/2026",
                    "status"          => "Pending Review"
                ],
                [
                    "application_id" => "#BH1048",
                    "amount"         => 50000,
                    "date"           => "23/06/2025",
                    "status"          => "Pending Review"
                ]
            ];
            return $history;
        }
    }

    public function get_loan_details($user_id){
        if ($_SESSION['role']==='user') {
            # Use sql request if database
            $current_loan = [
                [
                    "application_id" => "#AI82741",
                    "type"           => "Official Loan",
                    "amount"         => 2000,
                    "start_date"           => "05/04/2026",
                    "status"          => "Pending",
                    "remaining"      => 2000,
                    "duration_months"      => 63,
                    "monthly_payment"      => 32,
                    "interest_rate"      => 1.6

                ],
                [
                    "application_id" => "#BH1048",
                    "type"           => "Official Loan",
                    "amount"         => 50000,
                    "start_date"           => "23/06/2025",
                    "status"          => "Approved",
                    "remaining"      => 1744,
                    "duration_months"      => 63,
                    "monthly_payment"      => 32,
                    "interest_rate"      => 1.6
                ]
            ];
            return $current_loan;
        }
    }

    public function get_simulation_details($id){
        # Use sql request if database
        if ($_SESSION['role']==='user') {
            if ($id==='#BQ2834') {
            $sim = 
                [
                    "application_id" => "#BQ2834",
                    "type"           => "Simulation",
                    "amount"         => 2000,
                    "date"           => "04/04/2026",
                    "status"          => "Approved",
                    "duration_months"      => 63,
                    "interest_rate"      => 1.6,
                    "risk_score"      => 12
            ];
            } else {
                $sim = 
                [ 
                "application_id" => "#BIA1224",
                "type"           => "Simulation",
                "amount"         => 4000,
                "date"           => "04/04/2026",
                "status"          => "Refused",
                "duration_months"      => 63,
                "interest_rate"      => 1.6,
                "risk_score"      => 54
            ];
            }
            return $sim;
        }
    }
}