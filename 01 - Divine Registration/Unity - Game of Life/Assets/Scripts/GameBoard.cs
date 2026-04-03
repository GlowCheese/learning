using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

public class GameBoard : MonoBehaviour
{
    [SerializeField] private float updateInterval = 0.05f;

    [SerializeField] private Tilemap currState;
    [SerializeField] private Tilemap nextState;

    [SerializeField] private Tile aliveTile;
    [SerializeField] private Pattern pattern;

    private HashSet<Vector3Int> aliveCells;
    private Dictionary<Vector3Int, int> neighborCnt;

    private void Awake()
    {
        aliveCells = new HashSet<Vector3Int>();
        neighborCnt = new Dictionary<Vector3Int, int>();
    }

    private void Start()
    {
        SetPattern(pattern);
    }

    private void SetPattern(Pattern pattern)
    {
        Clear();

        Vector2Int center = pattern.GetCenter();

        for (int i = 0; i < pattern.cells.Length; i++)
        {
            Vector3Int cell = (Vector3Int)(pattern.cells[i] - center);
            currState.SetTile(cell, aliveTile);
            aliveCells.Add(cell);
        }
    }

    private void Clear()
    {
        currState.ClearAllTiles();
        nextState.ClearAllTiles();
    }

    private void OnEnable()
    {
        StartCoroutine(Simulate());
    }

    private IEnumerator Simulate()
    {
        var waitTime = new WaitForSeconds(updateInterval);
        while (enabled)
        {
            UpdateState();

            yield return waitTime;
        }
    }

    private void UpdateState()
    {
        neighborCnt.Clear();

        foreach (Vector3Int cell in aliveCells)
        {
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    Vector3Int adjCell = cell + new Vector3Int(x, y);
                    if (!neighborCnt.ContainsKey(adjCell))
                    {
                        neighborCnt[adjCell] = 0;
                    }

                    if (x != 0 || y != 0)
                    {
                        neighborCnt[adjCell] += 1;
                    }
                }
            }
        }

        foreach (var kv in neighborCnt)
        {
            Vector3Int cell = kv.Key;
            int cnt = kv.Value;

            if (!IsAlive(cell) && cnt == 3)
            {
                nextState.SetTile(cell, aliveTile);
                aliveCells.Add(cell);
            }
            else if (IsAlive(cell) && (cnt < 2 || cnt > 3))
            {
                nextState.SetTile(cell, null);
                aliveCells.Remove(cell);
            }
            else
            {
                nextState.SetTile(cell, currState.GetTile(cell));
            }
        }

        var temp = currState;
        currState = nextState;
        nextState = temp;
        nextState.ClearAllTiles();
    }

    private bool IsAlive(Vector3Int cell)
    {
        return currState.HasTile(cell);
    }
}
