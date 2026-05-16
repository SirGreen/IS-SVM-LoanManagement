<?php

$id_to_label = [
    // Status of existing checking account
    'A11' => '< 0 DM',
    'A12' => '0 <= ... < 200 DM',
    'A13' => '>= 200 DM / salary assignments for at least 1 year',
    'A14' => 'no checking account',

    // Credit history
    'A30' => 'no credits taken/ all credits paid back duly',
    'A31' => 'all credits at this bank paid back duly',
    'A32' => 'existing credits paid back duly till now',
    'A33' => 'delay in paying off in the past',
    'A34' => 'critical account/ other credits existing (not at this bank)',

    // Purpose
    'A40' => 'car (new)',
    'A41' => 'car (used)',
    'A42' => 'furniture/equipment',
    'A43' => 'radio/television',
    'A44' => 'domestic appliances',
    'A45' => 'repairs',
    'A46' => 'education',
    'A48' => 'retraining',
    'A49' => 'business',
    'A410' => 'others',

    // Savings account/bonds
    'A61' => '< 100 DM',
    'A62' => '100 <= ... < 500 DM',
    'A63' => '500 <= ... < 1000 DM',
    'A64' => '>= 1000 DM',
    'A65' => 'unknown/ no savings account',

    // Present employment since
    'A71' => 'unemployed',
    'A72' => '< 1 year',
    'A73' => '1 <= ... < 4 years',
    'A74' => '4 <= ... < 7 years',
    'A75' => '>= 7 years',

    // Personal status and sex
    'A91' => 'male : divorced/separated',
    'A92' => 'female : divorced/separated/married',
    'A93' => 'male : single',
    'A94' => 'male : married/widowed',
    'A95' => 'female : single',

    // Other debtors / guarantors
    'A101' => 'no other debtors/guarantors',
    'A102' => 'co-applicant',
    'A103' => 'guarantor',

    // Property
    'A121' => 'real estate',
    'A122' => 'building society savings agreement/ life insurance',
    'A123' => 'car or other',
    'A124' => 'unknown / no property',

    // Other installment plans
    'A141' => 'bank',
    'A142' => 'stores',
    'A143' => 'no other installment plans',

    // Housing
    'A151' => 'rent',
    'A152' => 'own',
    'A153' => 'for free',

    // Job
    'A171' => 'unemployed/ unskilled - non-resident',
    'A172' => 'unskilled - resident',
    'A173' => 'skilled employee / official',
    'A174' => 'management/ self-employed/ highly qualified employee/ officer',

    // Telephone
    'A191' => 'no telephone',
    'A192' => 'yes, registered under the customers name',

    // Foreign worker
    'A201' => 'yes',
    'A202' => 'no',
];

$label_to_id = [
    // Status of existing checking account
    '< 0 DM'                                                        => 'A11',
    '0 <= ... < 200 DM'                                             => 'A12',
    '>= 200 DM / salary assignments for at least 1 year'            => 'A13',
    'no checking account'                                           => 'A14',

    // Credit history
    'no credits taken/ all credits paid back duly'                  => 'A30',
    'all credits at this bank paid back duly'                       => 'A31',
    'existing credits paid back duly till now'                      => 'A32',
    'delay in paying off in the past'                               => 'A33',
    'critical account/ other credits existing (not at this bank)'   => 'A34',

    // Purpose
    'car (new)'                                                     => 'A40',
    'car (used)'                                                    => 'A41',
    'furniture/equipment'                                           => 'A42',
    'radio/television'                                              => 'A43',
    'domestic appliances'                                           => 'A44',
    'repairs'                                                       => 'A45',
    'education'                                                     => 'A46',
    'retraining'                                                    => 'A48',
    'business'                                                      => 'A49',
    'others'                                                        => 'A410',

    // Savings account/bonds
    '< 100 DM'                                                      => 'A61',
    '100 <= ... < 500 DM'                                           => 'A62',
    '500 <= ... < 1000 DM'                                          => 'A63',
    '>= 1000 DM'                                                    => 'A64',
    'unknown/ no savings account'                                   => 'A65',

    // Present employment since
    'unemployed'                                                    => 'A71',
    '< 1 year'                                                      => 'A72',
    '1 <= ... < 4 years'                                            => 'A73',
    '4 <= ... < 7 years'                                            => 'A74',
    '>= 7 years'                                                    => 'A75',

    // Personal status and sex
    'male : divorced/separated'                                     => 'A91',
    'female : divorced/separated/married'                           => 'A92',
    'male : single'                                                 => 'A93',
    'male : married/widowed'                                        => 'A94',
    'female : single'                                               => 'A95',

    // Other debtors / guarantors
    'no other debtors/guarantors'                                   => 'A101',
    'co-applicant'                                                  => 'A102',
    'guarantor'                                                     => 'A103',

    // Property
    'real estate'                                                   => 'A121',
    'building society savings agreement/ life insurance'            => 'A122',
    'car or other'                                                  => 'A123',
    'unknown / no property'                                         => 'A124',

    // Other installment plans
    'bank'                                                          => 'A141',
    'stores'                                                        => 'A142',
    'no other installment plans'                                    => 'A143',

    // Housing
    'rent'                                                          => 'A151',
    'own'                                                           => 'A152',
    'for free'                                                      => 'A153',

    // Job
    'unemployed/ unskilled - non-resident'                          => 'A171',
    'unskilled - resident'                                          => 'A172',
    'skilled employee / official'                                   => 'A173',
    'management/ self-employed/ highly qualified employee/ officer' => 'A174',

    // Telephone
    'no telephone'                                                  => 'A191',
    'yes, registered under the customers name'                      => 'A192',

    // Foreign worker
    'yes'                                                           => 'A201',
    'no'                                                            => 'A202',
];

$features_to_values = [
    'Status of existing checking account' => [
        '< 0 DM',
        '0 <= ... < 200 DM',
        '>= 200 DM / salary assignments for at least 1 year',
        'no checking account',
    ],
    'Credit history' => [
        'no credits taken/ all credits paid back duly',
        'all credits at this bank paid back duly',
        'existing credits paid back duly till now',
        'delay in paying off in the past',
        'critical account/ other credits existing (not at this bank)',
    ],
    'Purpose' => [
        'car (new)',
        'car (used)',
        'furniture/equipment',
        'radio/television',
        'domestic appliances',
        'repairs',
        'education',
        'retraining',
        'business',
        'others',
    ],
    'Savings account/bonds' => [
        '< 100 DM',
        '100 <= ... < 500 DM',
        '500 <= ... < 1000 DM',
        '>= 1000 DM',
        'unknown/ no savings account',
    ],
    'Present employment since' => [
        'unemployed',
        '< 1 year',
        '1 <= ... < 4 years',
        '4 <= ... < 7 years',
        '>= 7 years',
    ],
    'Personal status and sex' => [
        'male : divorced/separated',
        'female : divorced/separated/married',
        'male : single',
        'male : married/widowed',
        'female : single',
    ],
    'Other debtors / guarantors' => [
        'no other debtors/guarantors',
        'co-applicant',
        'guarantor',
    ],
    'Property' => [
        'real estate',
        'building society savings agreement/ life insurance',
        'car or other',
        'unknown / no property',
    ],
    'Other installment plans' => [
        'bank',
        'stores',
        'no other installment plans',
    ],
    'Housing' => [
        'rent',
        'own',
        'for free',
    ],
    'Job' => [
        'unemployed/ unskilled - non-resident',
        'unskilled - resident',
        'skilled employee / official',
        'management/ self-employed/ highly qualified employee/ officer',
    ],
    'Telephone' => [
        'no telephone',
        'yes, registered under the customers name',
    ],
    'Foreign worker' => [
        'yes',
        'no',
    ],
];

$old_name_to_modern_name = [
    '< 0 DM'                                                        => 'Empty Checking Account',
    '0 <= ... < 200 DM'                                             => 'Less than 200$',
    '>= 200 DM / salary assignments for at least 1 year'            => 'More than 200$',
    'no checking account'                                           => 'No Checking Account',
    '< 100 DM'                                                      => 'Less than 100$',
    '100 <= ... < 500 DM'                                           => 'Between 100 and 500$',
    '500 <= ... < 1000 DM'                                          => 'Between 500 and 1000$',
    '>= 1000 DM'                                                    => 'More than 1000$',
    'unknown/ no savings account'                                   => 'Unknown/No Savings Account',

]

?>