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
                $current_loan = 2;
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
                
            case 'details':
                include '../views/users/simulation_details.php';
                break;

            case 'loan_details':
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