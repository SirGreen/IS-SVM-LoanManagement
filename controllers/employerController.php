<?php
require_once '../models/user.php';
class EmployerController {
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
                $pending_list = $this->user_model->get_loan_pending();
                $stats = [
                    "pending_reviews" => 12,
                    "completed_this_month" => 45
                ];
                include '../views/employer/dashboard.php';
                break;

            case 'loan':
                include '../views/employer/loan_form.php';
                break;

            case 'apply':
                break;
                
            case 'details':
                break;

            case 'download':
                break;
            
            default:
                break;
        }
    }
}