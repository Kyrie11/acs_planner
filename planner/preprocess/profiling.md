# 预处理性能优化要点

下面这些优化已经体现在代码里，适合 nuPlan `.db + .gpkg`：

1. **SQLite 只读连接复用**
   - `mode=ro&immutable=1`
   - `query_only=ON`
   - 大页缓存 + `mmap_size`

2. **批量范围查询**
   - 用时间窗口拉取，而不是逐帧 / 逐 token 小 query

3. **prefix index 一次性生成**
   - 不在训练时反复从 log 数据里推前缀切片

4. **GeoPackage 直读 SQLite**
   - 绕开 `geopandas.read_file()` 的高启动成本
   - 仅提取必要 layer
   - 只保留 bounds / type / attrs 这类轻元数据用于过滤

5. **Map 元数据预编译**
   - 每个城市 `.gpkg` 只做一次离线转换
   - 运行期优先命中缓存，不反复解 WKB

6. **把 base anchor 候选与 action-conditioned anchor 分开**
   - base anchor graph 可按 `(scenario_token, iteration_index, route_hash)` 缓存
   - refined action 只在末端做 path-specific 过滤

## 若你现有代码特别卡，优先排查

- 有没有在 DataLoader worker 里每个样本重开 SQLite / map 文件
- 有没有在每个 prefix 上重复调用高层 map API 拉全量 lane objects
- 有没有把 geopandas 当随机查询引擎
- 有没有对所有 agents 做 lane association，而不是先裁剪 top-N
- 有没有在全动作集上 compile support
