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
}