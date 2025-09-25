def make_time_callback(distance_matrix, service_times, manager):
    def callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node] + service_times[from_node]
    return callback

def make_service_callback(service_times, manager):
    def callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return service_times[from_node]
    return callback
