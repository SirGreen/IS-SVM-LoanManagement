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
                $limit = 3;
                $total_rewiew = 3;
                $total_pages = ceil($total_rewiew / $limit);
                $current_page = isset($_GET['p']) ? (int)$_GET['p'] : 1;
                $pending_list = ($this->get_uncertain($current_page,$limit))['data'];
                include '../views/employer/dashboard.php';
                break;

            case 'loan':
                include '../views/employer/loan_form.php';
                break;

            case 'show_results':
                $fields = $_SESSION['form_data'] ?? [];
                $result = $_SESSION['api_results']['data'] ?? null;
                unset($_SESSION['form_data']);
                unset($_SESSION['api_results']);
                include '../views/employer/loan_form.php';
                break;

            case 'apply':
                $this->run_model();
                break;
                
            case 'review':
                break;

            case 'download':
                break;

            case 'run_model':
                $this->run_model();
                break;
            
            default:
                break;
        }
    }

    private function run_model(){
        $endpoint = "api/v1/predict";
        $data = $_POST;
        $jsonData = json_encode($data);
        $response = $this->api->call($endpoint,'GET',$jsonData);
        return $response;
    }

    private function get_uncertain($page,$limit){
        $endpoint = "api/v1/officer/uncertain";
        $response = $this->api->call($endpoint,'GET',['page'=>$page,'limit'=>$limit]);
        return $response;
    }
}