<?php

/*import UIKit
import Alamofire
import SwiftyJSON

let getSongInfoURL = "https://jeremylian.com/musicalchemy/service/getSongInfo.php"
        
        let params : [String : String] = ["songId" : String(song!.songId)]
        
        Alamofire.request(getSongInfoURL, method : .post , parameters : params).responseJSON{
            response in
            
            if response.result.isSuccess{
                //printing response
                print(response)
                
                let responseJson : JSON = JSON(response.result.value!)

                if responseJson["error"].boolValue == false {
                    self.songTitleLabel.text = "Title: \(responseJson["songInfo"].array![0]["song_songTitle"].stringValue)"
                    self.artistLabel.text = "Artist: \(responseJson["songInfo"].array![0]["userName"].stringValue)"
                }
                else{
                    self.songTitleLabel.text = "Unable able to load song info. Please try again."
                }
            }
            else{
                print("Error: \(String(describing: response.result.error))")
            }
        }    

func getSongsBySongTitle(text: String){
    let params : [String : String] = ["song_genre_theme" : text]
    
    Alamofire.request(getSongByTitleUrl, method : .post , parameters : params).responseJSON{
        response in
        
        if response.result.isSuccess{
            print(response)
            let responseJson : JSON = JSON(response.result.value!)
            self.parseSongsBySongTitleIntoObject(json: responseJson)
        }
        else{
            print("Error: \(String(describing: response.result.error))")
        }
    }
}
*/
    $track= 'Fix';
    $artist = 'Coldplay';
    echo $track;
    echo $artist;
    $tmp = exec("python3 /home/akshat/Downloads/trial.py $track $artist");
    // Decode the result
    $result = json_decode($tmp, true);

    var_dump($result);
    print_r($result);

?>