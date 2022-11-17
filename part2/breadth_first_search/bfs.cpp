#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define ALPHA 14
#define BETA 24

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int *mf,
    bool *bitmap,
    bool *new_bitmap)
{
    int sum = 0;
    int node, start_edge, end_edge;
    int outgoing;
    int index;

    #pragma omp parallel for reduction(+:sum) schedule(static, 1) private(node,start_edge,end_edge,outgoing,index)
    for (int i = 0; i < frontier->count; i++)
    {
        node = frontier->vertices[i]; // 目前在frontier的哪個vertic上
        start_edge = g->outgoing_starts[node];
        end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            outgoing = g->outgoing_edges[neighbor];
            if(distances[outgoing] == NOT_VISITED_MARKER){
                if (__sync_bool_compare_and_swap(distances+outgoing, NOT_VISITED_MARKER, distances[node]+1)) {
                    index = __sync_fetch_and_add(&(new_frontier->count), 1);
                    new_frontier->vertices[index] = outgoing;
                    new_bitmap[outgoing] = 1;
                    sum += outgoing_size(g, outgoing);
                }
            }
        }
    }
    *mf = sum;
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int *mf,
    bool *bitmap,
    bool *new_bitmap)
{
    int sum = 0;
    int count = 0;
    int start_edge, end_edge;
    int incoming;

    #pragma omp parallel for reduction(+:sum,count) schedule(static, 1) private(start_edge,end_edge,incoming)
    for (int v=0; v<g->num_nodes; v++) {
        if (distances[v] == NOT_VISITED_MARKER) {
            start_edge = g->incoming_starts[v];
            end_edge = (v == g->num_nodes-1)
                            ? g->num_edges
                            : g->incoming_starts[v+1];
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                incoming = g->incoming_edges[neighbor];
                if(bitmap[incoming]){ 
                    distances[v] = distances[incoming] + 1;
                    new_bitmap[v] = 1;
                    new_frontier->vertices[v] = 1;
                    sum += outgoing_size(g, v);
                    count++;
                    break;
                }
            }
        }
    }
    new_frontier->count += count;
    *mf = sum;
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    int mf = 0;

    bool *bitmap, *new_bitmap;
    bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));
    new_bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));
    bitmap[0] = 1;
    new_bitmap[0] = 1;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances, &mf, bitmap, new_bitmap);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        bool *tmp_b = new_bitmap;
        new_bitmap = bitmap;
        bitmap = tmp_b;

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    free(bitmap);
    free(new_bitmap);
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    int mf = 0;

    bool *bitmap, *new_bitmap;
    bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));
    new_bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));
    bitmap[0] = 1;
    new_bitmap[0] = 1;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    sol->distances[ROOT_NODE_ID] = 0;
    frontier->count++; // to get into for loop

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        bottom_up_step(graph, frontier, new_frontier, sol->distances, &mf, bitmap, new_bitmap); // handle one layer of frontier

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        bool *tmp_b = new_bitmap;
        new_bitmap = bitmap;
        bitmap = tmp_b;

        vertex_set *tmp_f = new_frontier;
        new_frontier = frontier;
        frontier = tmp_f;
    }
    free(bitmap);
    free(new_bitmap);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    int mf = 0;
    int nf = 0;
    int mu = graph->num_edges;
    bool mode = true;
    bool isGrowing = true;
    int preCnt = 0;

    bool *bitmap, *new_bitmap;
    bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));
    new_bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));
    bitmap[0] = 1;
    new_bitmap[0] = 1;

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    do {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        preCnt = frontier->count;

        vertex_set_clear(new_frontier);
        for (int i = 0; i < graph->num_nodes; i++) {
            new_frontier->vertices[i] = NOT_VISITED_MARKER;
        }

        if (mode) {
            top_down_step(graph, frontier, new_frontier, sol->distances, &mf, bitmap, new_bitmap);
        }
        else {
            bottom_up_step(graph, frontier, new_frontier, sol->distances, &mf, bitmap, new_bitmap);
        }

        // determine next time mode
        isGrowing = (preCnt < new_frontier->count);
        nf = new_frontier->count;
        mu -= mf;

        vertex_set_clear(frontier);
        for (int i = 0; i < graph->num_nodes; i++) {
            frontier->vertices[i] = NOT_VISITED_MARKER;
        }

        if (mode & (mf > (float)mu/ALPHA) & isGrowing) {
            mode = false; // top-down -> buttom-up

            for (int i=0; i<new_frontier->count; i++) {
                frontier->vertices[new_frontier->vertices[i]] = 1;
                frontier->count++;
            }
        }
        else if (!mode & (nf < (float)graph->num_edges/BETA) & !isGrowing) {
            mode = true;  // buttom-up -> top-down

            for (int i=0; i<graph->num_nodes; i++) {
                if (new_frontier->vertices[i] != NOT_VISITED_MARKER) {
                    frontier->vertices[frontier->count] = i;
                    frontier->count++;
                }
            }
        }
        else {
            vertex_set *tmp_f = new_frontier;
            new_frontier = frontier;
            frontier = tmp_f;
        }

        bool *tmp_b = new_bitmap;
        new_bitmap = bitmap;
        bitmap = tmp_b;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    } while (frontier->count != 0);

    free(bitmap);
    free(new_bitmap);
}
