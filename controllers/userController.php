<?php
require_once '../models/user.php';

class UserController {
    private $api;
    private $user_model;

    public function __construct() {
        $this->api = new APIService();
        $this->user_model = new User();
    }
    
    public function handle_request($action) {
        include __DIR__ . '/../views/header.php';
        switch ($action) {
            case 'dashboard':
                $current_loan = $this->user_model->get_loan_number($_SESSION['user_id']);
                $request_history = $this->user_model->get_request_history($_SESSION['user_id']);
                include '../views/users/dashboard.php';
                break;

            case 'simulation':
                $official_loan = false;
                include '../views/users/loan_form.php';
                break;

            case 'apply':
                $official_loan = true;
                include '../views/users/loan_form.php';
                break;

            case 'show_results':
                $fields = $_SESSION['form_data'] ?? [];
                $result = $_SESSION['api_results']['data'] ?? null;
                unset($_SESSION['form_data']);
                unset($_SESSION['api_results']);
                include '../views/users/loan_form.php';
                break;
                
            case 'details':
                $id = $_GET['id'];
                $simulation_details = $this->user_model->get_simulation_details($id);
                // On récupère les données de la simulation avec des valeurs par défaut au cas où
                $sim_id = $simulation_details['application_id'] ?? '#000000';
                $amount = $simulation_details['amount'] ?? 0;
                $duration = $simulation_details['duration_months'] ?? 0;
                $rate = $simulation_details['interest_rate'] ?? 0;
                $date = $simulation_details['date'] ?? 'N/A';
                $status = $simulation_details['status'] ?? 'Approved';
                $risk_score = $simulation_details['risk_score'] ?? 0;

                // Variables pour le style selon le résultat de l'IA
                $is_approved = ($status === 'Approved');
                $status_title = $is_approved ? 'Simulation Approved' : 'Simulation Refused';
                $status_color = $is_approved ? 'text-success' : 'text-danger';
                $badge_class = $is_approved ? 'bg-success text-white' : 'bg-danger text-white';

                // Détermination de la couleur de la jauge de risque
                if ($risk_score < 30) {
                    $risk_class = 'bg-success';
                    $risk_text = 'Low Risk';
                } elseif ($risk_score < 60) {
                    $risk_class = 'bg-warning text-dark';
                    $risk_text = 'Medium Risk';
                } else {
                    $risk_class = 'bg-danger';
                    $risk_text = 'High Risk';
                }
                include '../views/users/simulation_details.php';
                break;

            case 'loan_details':
                $current_loan = $this->user_model->get_loan_details($_SESSION['user_id']);
                include '../views/users/loan_details.php';
                break;

            case 'download':
                break;
            
            default:
                break;
        }
    }
}
?>