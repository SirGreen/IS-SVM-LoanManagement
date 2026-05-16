<?php
class WorkshopController {
    private $api;

    public function __construct() {
        $this->api = new APIService();
    }

    public function handle_request($action) {
        include __DIR__ . '/../views/header.php';
        switch ($action) {
            case 'dashboard':
                $stats = $this->get_stats()['data']['stats'];
                include '../views/data_scientist/model_workshop.php';
                break;

            case 'update_margin':
                $response = $this->set_margin();
                $_SESSION['flash'] = "Margin updated successfully !";
                $_SESSION['flash_type'] = "success";
                header("Location: index.php?page=model_workshop&action=dashboard");
                exit();
                break;
                
            case 'train':
                $response = $this->train();
                $_SESSION['flash'] = "Training has started !";
                $_SESSION['flash_type'] = "success";
                header("Location: index.php?page=model_workshop&action=dashboard");
                exit();
                break;
            
            default:
                break;
        }
    }
    

    public function train() {
        if ($_SERVER['REQUEST_METHOD'] === 'POST') {
            $endpoint = "api/v1/admin/retrain";
            $api_key = "api_key";
            // $data = $_POST;
            $file = $_FILES['dataset'];
            $fileName    = $file['name'];
            $fileTmpPath = $file['tmp_name'];
            $uploadFolder = __DIR__ . '/../data/';
            if (!is_dir($uploadFolder)) {
                mkdir($uploadFolder, 0777, true);
            }
    
            $destination = $uploadFolder . $fileName;
            if (move_uploaded_file($fileTmpPath, $destination)) {
                $data = ['dataset_zip_path'=>$destination];
                $json_data = json_encode($data);
                $response = $this->api->call($endpoint,'POST',$json_data,$api_key);
                return $response;     
            }     
        }
    }

    private function get_margin(){
        $endpoint = "api/v1/admin/uncertainty-margin";
        $api_key = "api_key";
        $response = $this->api->call($endpoint,'GET',null,$api_key);
        return $response;
    }

    private function set_margin(){
        $data = ['uncertainty_margin' => $_POST['margin']];
        $endpoint = "api/v1/admin/uncertainty-margin";
        $api_key = "api_key";
        $json_data = json_encode($data);
        $response = $this->api->call($endpoint,'POST',$json_data,$api_key);
        return $response;
    }

    private function get_stats(){
        $endpoint = "api/v1/admin/stats";
        $api_key = "api_key";
        $response = $this->api->call($endpoint,'GET',null,$api_key);
        return $response;
    }
}