from flask import Flask, request, render_template, jsonify
import keras
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Loading the pickled model
model_file_path = 'model.pkl'
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)

#Loading pca
pca_file_path = 'reduced_features.pkl'
with open(pca_file_path, 'rb') as pca_file:
    pca = pickle.load(pca_file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    # Preprocess data if needed
    def preprocess_input_data(input_data):
        scaled_column_names = [
            'land', 'logged_in', 'is_host_login', 'is_guest_login', 'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp',
            'service_IRC', 'service_X11', 'service_Z39_50', 'service_aol', 'service_auth', 'service_bgp', 'service_courier',
            'service_csnet_ns', 'service_ctf', 'service_daytime', 'service_discard', 'service_domain', 'service_domain_u',
            'service_echo', 'service_eco_i', 'service_ecr_i', 'service_efs', 'service_exec', 'service_finger', 'service_ftp',
            'service_ftp_data', 'service_gopher', 'service_harvest', 'service_hostnames', 'service_http', 'service_http_2784',
            'service_http_443', 'service_http_8001', 'service_imap4', 'service_iso_tsap', 'service_klogin', 'service_kshell',
            'service_ldap', 'service_link', 'service_login', 'service_mtp', 'service_name', 'service_netbios_dgm',
            'service_netbios_ns', 'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp', 'service_ntp_u',
            'service_other', 'service_pm_dump', 'service_pop_2', 'service_pop_3', 'service_printer', 'service_private',
            'service_red_i', 'service_remote_job', 'service_rje', 'service_shell', 'service_smtp', 'service_sql_net',
            'service_ssh', 'service_sunrpc', 'service_supdup', 'service_systat', 'service_telnet', 'service_tftp_u',
            'service_tim_i', 'service_time', 'service_urh_i', 'service_urp_i', 'service_uucp', 'service_uucp_path',
            'service_vmnet', 'service_whois', 'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0',
            'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH'
        ]
        custom_data=pd.DataFrame()
        for col in scaled_column_names:
            if col in input_data.keys():
                custom_data[col]=[input_data[col]]
        #custom_data
        
        # Defining the list of column names
        column_names = [
            'protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp',
            'service_IRC', 'service_X11', 'service_Z39_50', 'service_aol', 'service_auth', 'service_bgp',
            'service_courier', 'service_csnet_ns', 'service_ctf', 'service_daytime', 'service_discard',
            'service_domain', 'service_domain_u', 'service_echo', 'service_eco_i', 'service_ecr_i', 'service_efs',
            'service_exec', 'service_finger', 'service_ftp', 'service_ftp_data', 'service_gopher', 'service_harvest',
            'service_hostnames', 'service_http', 'service_http_2784', 'service_http_443', 'service_http_8001',
            'service_imap4', 'service_iso_tsap', 'service_klogin', 'service_kshell', 'service_ldap', 'service_link',
            'service_login', 'service_mtp', 'service_name', 'service_netbios_dgm', 'service_netbios_ns',
            'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp', 'service_ntp_u',
            'service_other', 'service_pm_dump', 'service_pop_2', 'service_pop_3', 'service_printer', 'service_private',
            'service_red_i', 'service_remote_job', 'service_rje', 'service_shell', 'service_smtp', 'service_sql_net',
            'service_ssh', 'service_sunrpc', 'service_supdup', 'service_systat', 'service_telnet', 'service_tftp_u',
            'service_tim_i', 'service_time', 'service_urh_i', 'service_urp_i', 'service_uucp', 'service_uucp_path',
            'service_vmnet', 'service_whois',
            'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3',
            'flag_SF', 'flag_SH'
        ]

        # Creating an empty DataFrame with the specified columns
        custom_data_df = pd.DataFrame(columns=column_names)

        # Adding the initial row of zeros
        initial_row = pd.Series(np.zeros(len(column_names)), index=column_names)
        custom_data_df = custom_data_df.append(initial_row, ignore_index=True)

        # Updating 'protocol_type' columns
        custom_data_df['protocol_type_'+input_data['protocol_type']] = 1

        # Updating 'service' columns
        custom_data_df['service_' + input_data['service']] = 1

        # Updating 'flag' columns
        custom_data_df['flag_' + input_data['flag']] = 1
        
        custom_data=pd.concat([custom_data, custom_data_df], axis=1)
        custom_data_reduced=pca.transform(custom_data)
        return custom_data_reduced
    ################################
    
    processed_data=preprocess_input_data(data)

    predicted_probabilities = model.predict(processed_data)
    outcome_mapping={'normal': 0, 'neptune': 1, 'warezclient': 2, 'ipsweep': 3, 
                     'portsweep': 4,'teardrop': 5,'nmap': 6,'satan': 7,'smurf': 8,
                     'pod': 9,'back': 10,'guess_passwd': 11,'ftp_write': 12,
                     'multihop': 13,'rootkit': 14,'buffer_overflow': 15,'imap': 16,
                     'warezmaster': 17,'phf': 18,'land': 19,'loadmodule': 20,
                     'spy': 21,'perl': 22}
    # Get the predicted class label (index with highest probability)
    predicted_class_index = np.argmax(predicted_probabilities)

    # Inverse mapping to get the class label string
    predicted_class = list(outcome_mapping.keys())[predicted_class_index]
    return jsonify({"Prediction": predicted_class})

if __name__ == "__main__":
    app.run()
