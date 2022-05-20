import time
import numpy as np
from torch_geometric.typing import Size
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import networkx as nx
from torch_scatter import scatter
from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch
import collections
# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model
from env.env_utils import override


def MinimalJobshopSat(data):
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = data
    n_j = len(jobs_data)
    n_m = len(jobs_data[0])

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1
    status = solver.Solve(model)

    # Create one list of assigned tasks per machine.
    assigned_jobs = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            assigned_jobs[machine].append(
                assigned_task_type(
                    start=solver.Value(all_tasks[job_id, task_id].start),
                    job=job_id,
                    index=task_id,
                    duration=task[1]))

    # Create per machine output lines.
    machine_assign_mat = []
    for machine in all_machines:
        # Sort by starting time.
        assigned_jobs[machine].sort()
        for assigned_task in assigned_jobs[machine]:
            machine_assign_mat.append(assigned_task.job)

    if status == cp_model.OPTIMAL:
        return [0, solver.ObjectiveValue()], np.array(machine_assign_mat).reshape((n_m, n_j))
    elif status == cp_model.FEASIBLE:
        return [1, solver.ObjectiveValue()], np.array(machine_assign_mat).reshape((n_m, n_j))
    else:
        print('Not found any Sol. Return [-1, -1]')
        return [-1, -1], None


def exact_solver(instance):
    """
    instance: [n, 2, j, m]
    """

    ortools_makespan = []
    solutions = []
    for i, inst in enumerate(instance):
        print('CP-SAT solving instance:', i + 1)
        times_rearrange = np.expand_dims(inst[0], axis=-1)
        machines_rearrange = np.expand_dims(inst[1], axis=-1)
        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
        val, sol = MinimalJobshopSat(data.tolist())
        ortools_makespan.append(val[1])
        solutions.append(sol)

    return np.array(ortools_makespan), solutions


class ForwardPass(MessagePassing):
    def __init__(self, **kwargs):
        super(ForwardPass, self).__init__(**kwargs)

    def forward(self, g, size: Size = None, num_nodes_per_example=None):
        """
        g: torch_geometric.data.batch.Batch, g.dur is the duration; batch of disjunctive graphs.
        """
        device = g.dur.device
        duration = g.dur[:, 0]
        total_num_nodes = duration.shape[0]
        if num_nodes_per_example is None:
            num_nodes_per_example = scatter(torch.ones_like(g.batch), g.batch)
        index_S = torch.cumsum(num_nodes_per_example, dim=0) - num_nodes_per_example
        index_T = torch.cumsum(num_nodes_per_example, dim=0) - 1
        est = torch.zeros(size=[total_num_nodes, 1], dtype=torch.float32, device=device)
        mask_est = torch.ones(size=[total_num_nodes, 1], dtype=torch.int8, device=device)
        mask_est[index_S] = 0
        x = torch.cat([mask_est, est], dim=1)
        _track_visited = 1 - mask_est.clone().squeeze()
        fwd_topo_batches = [index_S]

        _count = 0
        while True:
            if x[:, 0].sum() == 0:
                break
            x = self.propagate(g.edge_index, x=x, dur=duration, size=size)
            top_batch = torch.where((x[:, 0] == 0) * (_track_visited == 0))[0]  # nodes with x[:, 0] == 0 at each itr
            fwd_topo_batches.append(top_batch)
            _track_visited[top_batch] = 1
            _count += 1

        obj = torch.index_select(x, 0, index_T)[:, 1]
        return x[:, 1], obj, _count, fwd_topo_batches

    @override  # override for arguments
    def message(self, x_j: Tensor, dur: Tensor, edge_index) -> Tensor:
        dur_j = dur[edge_index[0]]
        x_j[:, 1] = dur_j.masked_fill(x_j[:, 0].bool(), 0) + x_j[:, 1]
        return x_j


class BackwardPass(MessagePassing):
    def __init__(self, **kwargs):
        super(BackwardPass, self).__init__(**kwargs)

    def forward(self, g, make_span, size: Size = None, num_nodes_per_example=None, est=None):
        """
        g: torch_geometric.data.batch.Batch, g.dur is the duration; batch of disjunctive graphs.
        make_span: [n instance, ]
        """
        device = g.dur.device
        duration = g.dur[:, 0]
        total_num_nodes = duration.shape[0]
        if num_nodes_per_example is None:
            num_nodes_per_example = scatter(torch.ones_like(g.batch), g.batch)
        index_T = torch.cumsum(num_nodes_per_example, dim=0) - 1
        # add self-loop for T's since T's has no nei, fea_T = 0 after each propagate
        edge_index_with_self_loop = torch.cat([g.edge_index, torch.tile(index_T, (2, 1))], dim=1)
        lst = torch.zeros(size=[total_num_nodes, 1], dtype=torch.float32, device=device)
        lst[index_T] = - make_span.unsqueeze(1)
        lst_mask = torch.ones(size=[total_num_nodes, 1], dtype=torch.int8, device=device)
        lst_mask[index_T] = 0
        x = torch.cat([lst_mask, lst], dim=1)
        _track_visited = 1 - lst_mask.clone().squeeze()
        bwd_topo_batches = [index_T]
        _count = 0
        while True:
            if x[:, 0].sum() == 0:
                break
            x = self.propagate(edge_index_with_self_loop, x=x, dur=duration, size=size)
            top_batch = torch.where((x[:, 0] == 0) * (_track_visited == 0))[0]  # nodes with x[:, 0] == 0 at each itr
            bwd_topo_batches.append(top_batch)
            _track_visited[top_batch] = 1
            _count += 1
        return torch.abs(x[:, 1]), _count, bwd_topo_batches

    @override  # override for arguments
    def message(self, x_j: Tensor, dur: Tensor, edge_index) -> Tensor:
        dur_j = dur.squeeze()[edge_index[0]]
        x_j[:, 1] = dur_j.masked_fill(x_j[:, 0].bool(), 0) + x_j[:, 1]
        return x_j


class MassagePassingEval:
    def __init__(self):
        self.forward_pass = ForwardPass(aggr='max', flow="source_to_target")
        self.backward_pass = BackwardPass(aggr='max', flow="target_to_source")

    def eval(self, g, num_nodes_per_example=None):
        """
        g: torch_geometric.data.batch.Batch, g.dur is the duration; batch of disjunctive graphs.
        """
        est, make_span, fwd_count, fwd_topo_batches = self.forward_pass(g, num_nodes_per_example=num_nodes_per_example)
        lst, bwd_count, bwd_topo_batches = self.backward_pass(g, make_span, num_nodes_per_example=num_nodes_per_example, est=est)
        assert fwd_count == bwd_count
        return est, lst, make_span, fwd_count, fwd_topo_batches, bwd_topo_batches


def processing_order_to_edge_index(order, instance):
    """
    order: [n_m, n_j] a numpy array specifying the processing order on each machine, each row is a machine
    instance: [1, n_j, n_m] an instance as numpy array
    RETURN: edge index: [2, n_j * n_m +2] tensor for the directed disjunctive graph
    """
    dur, mch = instance[0], instance[1]
    n_j, n_m = dur.shape[0], dur.shape[1]
    n_opr = n_j*n_m

    adj = np.eye(n_opr, k=-1, dtype=int)  # Create adjacent matrix for precedence constraints
    adj[np.arange(start=0, stop=n_opr, step=1).reshape(n_j, -1)[:, 0]] = 0  # first column does not have upper stream conj_nei
    adj = np.pad(adj, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
    adj[[i for i in range(1, n_opr + 2 - 1, n_m)], 0] = 1  # connect S with 1st operation of each job
    adj[-1, [i for i in range(n_m, n_opr + 2 - 1, n_m)]] = 1  # connect last operation of each job to T
    adj = np.transpose(adj)

    # rollout ortools solution
    steps_basedon_sol = []
    for i in range(n_m):
        get_col_position_unsorted = np.argwhere(mch == (i+1))  # if m iterate from 1 then mch == (i+1)
        get_col_position_sorted = get_col_position_unsorted[order[i]]
        sol_i = order[i] * n_m + get_col_position_sorted[:, 1]
        steps_basedon_sol.append(sol_i.tolist())

    for operations in steps_basedon_sol:
        for i in range(len(operations) - 1):
            adj[operations[i]+1][operations[i+1]+1] += 1

    return torch.nonzero(torch.from_numpy(adj)).t().contiguous()


def cpm_forward(graph, topological_order=None):  # graph is a nx.DiGraph;
    # assert (graph.in_degree(topological_order[0]) == 0)
    earliest_ST = dict.fromkeys(graph.nodes, -float('inf'))
    if topological_order is None:
        topo_order = list(nx.topological_sort(graph))
    else:
        topo_order = topological_order
    earliest_ST[topo_order[0]] = 0.
    for n in topo_order:
        for s in graph.successors(n):
            if earliest_ST[s] < earliest_ST[n] + graph.edges[n, s]['weight']:
                earliest_ST[s] = earliest_ST[n] + graph.edges[n, s]['weight']
    # return is a dict where key is each node's ID, value is the length from source node s
    return earliest_ST


def cpm_backward(graph, makespan, topological_order=None):
    if topological_order is None:
        reverse_order = list(reversed(list(nx.topological_sort(graph))))
    else:
        reverse_order = list(reversed(topological_order))
    latest_ST = dict.fromkeys(graph.nodes, float('inf'))
    latest_ST[reverse_order[0]] = float(makespan)
    for n in reverse_order:
        for p in graph.predecessors(n):
            if latest_ST[p] > latest_ST[n] - graph.edges[p, n]['weight']:
                # assert latest_ST[n] - graph.edges[p, n]['weight'] >= 0, 'latest start times should is negative, BUG!'  # latest start times should be non-negative
                latest_ST[p] = latest_ST[n] - graph.edges[p, n]['weight']
    return latest_ST


def cpm_forward_and_backward(G):
    # calculate topological order
    t1 = time.time()
    topological_order = list(nx.topological_sort(G))
    print(time.time() - t1)
    # print(topological_order)
    # forward and backward pass
    earliest_start_time = np.fromiter(cpm_forward(graph=G, topological_order=topological_order).values(), dtype=np.float32)
    latest_start_time = np.fromiter(cpm_backward(graph=G, topological_order=topological_order, makespan=earliest_start_time[-1]).values(), dtype=np.float32)
    # assert np.where(earliest_start_time > latest_start_time)[0].shape[0] == 0, 'latest starting time is smaller than earliest starting time, bug!'  # latest starting time should be larger or equal to earliest starting time
    return earliest_start_time, latest_start_time, earliest_start_time[-1]


def topological_sort_grouped(G):
    t2 = time.time()
    print('yes')
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    zero_indegree = [v for v, d in G.in_degree() if d == 0]
    while zero_indegree:
        yield zero_indegree
        new_zero_indegree = []
        for v in zero_indegree:
            for _, child in G.edges(v):
                indegree_map[child] -= 1
                if not indegree_map[child]:
                    new_zero_indegree.append(child)
        new_zero_indegree.sort()
        zero_indegree = new_zero_indegree
    print(time.time() - t2)


if __name__ == "__main__":
    from generateJSP import uni_instance_gen

    ## test against ortools
    j, m, batch_size = {'low': 100, 'high': 101}, {'low': 20, 'high': 21}, 1
    l = 1
    h = 99
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)

    insts = [np.concatenate(
        [uni_instance_gen(n_j=np.random.randint(**j), n_m=np.random.randint(**m), low=l, high=h)]
    ) for _ in range(batch_size)]

    ortools_makespan, sol = exact_solver(insts)

    pygs = []
    eva = MassagePassingEval()
    for i, inst in enumerate(insts):
        print('Compute Cmax of sol of instance using message-passing:', i+1)
        edg_idx = processing_order_to_edge_index(order=sol[i], instance=inst)
        dur = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1)
        pygs.append(Data(dur=dur, edge_index=edg_idx, num_nodes=dur.shape[0]))
    pyg_batch = Batch.from_data_list(pygs).to(dev)
    est, lst, obj, count, fwd_batch, bwd_batch = eva.eval(pyg_batch)
    if np.array_equal(obj.squeeze().cpu().numpy(), ortools_makespan):
        print('message-passing evaluator get the same makespan when it rollouts ortools solution!')
    else:
        print('message-passing evaluator get the different makespan when it rollouts ortools solution.')


    from torch_geometric.utils import to_networkx, sort_edge_index
    pyg_batch.edge_index = sort_edge_index(pyg_batch.edge_index)
    pyg_batch.weight = pyg_batch.dur[pyg_batch.edge_index[0]]
    nxg = to_networkx(pyg_batch, edge_attrs=['weight'])
    ret = cpm_forward_and_backward(nxg)

    a = topological_sort_grouped(nxg)
    print(list(topological_sort_grouped(nxg)))
    print(torch.cat(fwd_batch).cpu().numpy())



