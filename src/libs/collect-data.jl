using CSV
using DataFrames
using Glob

@inline function process_files(input_directory, output_file)
    # ディレクトリ内のすべてのT_E_matome_all.csvファイルを検索
    files = glob("*/T_E_matome_all.csv", input_directory)
    println(files)

    # 出力用のデータフレームを作成
    output_df = DataFrame(N=[], a=[], b=[], c=[], d=[], B2=[], B3=[], B4=[], B5=[])

    for file in files
        # ファイルパスから必要な情報を抽出
        parts = split(file, '/')
        n, a, b, c, d = parse.(Int, parts[end-6:end-2])
        
        # CSVファイルを読み込む
        df = CSV.File(file) |> DataFrame

        # 出力用のデータフレームに追加
        push!(output_df, (n, a, b, c, d, df[2, 2], df[3, 2], df[4, 2], df[5, 2]))
    end

    # 結果をCSVファイルとして保存
    CSV.write(output_file, output_df, append=false)
end

@inline function test(input_directory, output_file)
    # ディレクトリ内のすべてのT_E_matome_all.csvファイルを検索
    files = glob("*/T_E_matome_all.csv", input_directory)
    println(files)

    # # 出力用のデータフレームを作成
    # output_df = DataFrame(N=[], a=[], b=[], c=[], d=[], B2=[], B3=[], B4=[], B5=[])

    # for file in files
    #     # ファイルパスから必要な情報を抽出
    #     parts = split(file, '/')
    #     n, a, b, c, d = parse.(Int, parts[end-6:end-2])
        
    #     # CSVファイルを読み込む
    #     df = CSV.File(file) |> DataFrame

    #     # 出力用のデータフレームに追加
    #     push!(output_df, (n, a, b, c, d, df[2, 2], df[3, 2], df[4, 2], df[5, 2]))
    # end

    # # 結果をCSVファイルとして保存
    # CSV.write(output_file, output_df, append=false)
end